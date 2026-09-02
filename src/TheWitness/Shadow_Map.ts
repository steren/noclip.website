import { GfxDevice, GfxFormat, GfxTexture, GfxTextureDimension, GfxTextureUsage, GfxTexFilterMode, GfxMipFilterMode, GfxWrapMode } from "../gfx/platform/GfxPlatform.js";
import { TextureMapping } from "../TextureHolder.js";
import { Asset_Type } from "./Assets.js";
import { TheWitnessGlobals } from "./Globals.js";
import { Entity_World } from "./Entity.js";

// What the game calls its shadow map is a height field: for every column of the world, the height
// of the tallest thing standing in it, top-down, over the whole island. It is the only description
// of the world's shape that is available all at once, which is what shadowing the sun needs -- a
// surface has to know about geometry that is nowhere near it, and may not even be loaded.
//
// The rectangle it covers is not written down anywhere in the data. These numbers come from fitting
// the field against the bounding boxes of the world-space meshes: the height it reports over a mesh
// has to be the height of that mesh. The 2048 and 4096 maps were fitted separately and agreed on
// the same rectangle to within half a percent, which is what says the fit is real.
const WORLD_X_AT_ROW_0 = 515.0;
const WORLD_Y_AT_COLUMN_0 = 518.4;
const WORLD_SIZE = 1082.8;

// A texel holds the tallest point in its footprint, so the ground reads as very slightly above
// itself and every lit surface would otherwise shadow-acne. Half a metre of slack clears it.
const DEPTH_BIAS = 0.5;
// The sun has an angular size, and the field is a metre or so per texel in any case; softening the
// comparison over a small height range stands in for a penumbra and hides the texel grid.
const PENUMBRA = 1.5;

const HEIGHT_MAP_SIZE = 2048;

export class Shadow_Map {
    public texture_mapping = new TextureMapping();
    public enabled = false;

    // The transform into the map, and back out of its 16-bit heights, as the shader wants them.
    public world_origin: [number, number] = [WORLD_Y_AT_COLUMN_0, WORLD_X_AT_ROW_0];
    public inv_world_size = 1.0 / WORLD_SIZE;
    public z_max = 0.0;
    public z_range = 1.0;
    public depth_bias = DEPTH_BIAS;
    public penumbra = PENUMBRA;

    private texture: GfxTexture | null = null;

    constructor(globals: TheWitnessGlobals, world: Entity_World) {
        const name = `${globals.entity_manager.universe_name}_shadow_map_${HEIGHT_MAP_SIZE}`;
        const bytes = globals.asset_manager.load_asset_bytes(Asset_Type.Texture, name);
        if (bytes === null) {
            console.warn(`TheWitness: no ${name}; the sun will not cast shadows`);
            return;
        }

        const view = bytes.createDataView();
        const width = view.getUint16(0x00, true), height = view.getUint16(0x02, true);
        const d3d_format = view.getUint32(0x1C, true);
        if (d3d_format !== 0x51) {
            console.warn(`TheWitness: ${name} is not the 16-bit height field we expect (format 0x${d3d_format.toString(16)})`);
            return;
        }

        // The field's own encoding: 0 is the top of the world, full scale is the bottom, so that
        // a column with nothing in it saturates.
        this.z_max = world.world_z_max;
        this.z_range = world.world_z_max - world.world_z_min;

        const heights = bytes.createTypedArray(Uint16Array, 0x20, width * height);
        const shadow = this.sweep(globals, heights, width, height);

        this.texture = globals.device.createTexture({
            dimension: GfxTextureDimension.n2D,
            width, height, depthOrArrayLayers: 1, numLevels: 1,
            pixelFormat: GfxFormat.U16_R_NORM,
            usage: GfxTextureUsage.Sampled,
        });
        globals.device.setResourceName(this.texture, `${name} (swept)`);
        globals.device.uploadTextureData(this.texture, 0, [shadow]);

        this.texture_mapping.gfxTexture = this.texture;
        this.texture_mapping.gfxSampler = globals.renderCache.createSampler({
            minFilter: GfxTexFilterMode.Bilinear,
            magFilter: GfxTexFilterMode.Bilinear,
            mipFilter: GfxMipFilterMode.Nearest,
            wrapS: GfxWrapMode.Clamp,
            wrapT: GfxWrapMode.Clamp,
        });
        this.enabled = true;
    }

    // Turn the height field into the height at which the sun arrives. Marching toward the sun per
    // fragment would answer the same question and cost a dozen taps to do it; sweeping the field
    // once along the sun instead answers it everywhere, and leaves the shader a single lookup.
    //
    // Reading along the sun's own direction, the shadow cast forward from everything behind falls
    // away at exactly the sun's slope, so each texel only has to look at the one upstream of it:
    //     shadow(p) = max(height(p), shadow(p - sun_step) - slope * step)
    private sweep(globals: TheWitnessGlobals, heights: Uint16Array, width: number, height: number): Uint16Array {
        const misc = globals.all_variables.misc;
        const sun_x = misc.sun_x as number, sun_y = misc.sun_y as number, sun_z = misc.sun_z as number;
        const sun_horizontal = Math.hypot(sun_x, sun_y);
        const slope = sun_z / sun_horizontal;

        // World +x runs along -row and world +y along -column, so the shadows travel this way.
        const du = sun_y / sun_horizontal, dv = sun_x / sun_horizontal;
        const units_per_texel = WORLD_SIZE / width;

        // Step along whichever axis the sun leans on, so that every texel is written exactly once
        // and reads a predecessor that has already been written.
        const along_rows = Math.abs(dv) >= Math.abs(du);
        const steps = along_rows ? height : width;
        const lanes = along_rows ? width : height;
        const lane_drift = along_rows ? du / dv : dv / du;
        const forward = (along_rows ? dv : du) > 0;
        const index = along_rows ? (s: number, l: number) => s * width + l : (s: number, l: number) => l * width + s;

        const fall = Math.hypot(1.0, lane_drift) * units_per_texel * slope;
        // Heights are stored top-down, so the sun eating into a shadow *raises* the stored value.
        const fall_encoded = (fall / this.z_range) * 65535.0;

        const shadow = new Uint16Array(width * height);
        const first = forward ? 0 : steps - 1;
        for (let l = 0; l < lanes; l++)
            shadow[index(first, l)] = heights[index(first, l)];

        const direction = forward ? 1 : -1;
        for (let k = 1; k < steps; k++) {
            const s = forward ? k : steps - 1 - k;
            const previous = s - direction;
            // The predecessor along the sun sits a fraction of a texel to the side; take it linearly.
            const drift = -direction * lane_drift;
            for (let l = 0; l < lanes; l++) {
                const source = l + drift;
                const l0 = Math.floor(source), t = source - l0, l1 = l0 + 1;
                let carried = 65535.0;
                if (l0 >= 0 && l1 < lanes)
                    carried = shadow[index(previous, l0)] * (1.0 - t) + shadow[index(previous, l1)] * t;
                else if (l0 >= 0 && l0 < lanes)
                    carried = shadow[index(previous, l0)];

                // min, because a smaller stored value is a greater height.
                const here = heights[index(s, l)];
                const decayed = carried + fall_encoded;
                shadow[index(s, l)] = Math.min(here, decayed < 65535.0 ? decayed : 65535.0);
            }
        }

        return shadow;
    }

    public destroy(device: GfxDevice): void {
        if (this.texture !== null)
            device.destroyTexture(this.texture);
    }
}
