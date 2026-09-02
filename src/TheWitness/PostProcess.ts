
// The Witness renders its scene in HDR and then puts it through exposure, bloom, a filmic curve
// and a vignette. The constants here are the game's own, out of tone_mapping.variables and
// bloom.variables; the code reads them from the world rather than repeating them.

import { GfxShaderLibrary } from "../gfx/helpers/GfxShaderLibrary.js";
import { fullscreenMegaState } from "../gfx/helpers/GfxMegaStateDescriptorHelpers.js";
import { makeBackbufferDescSimple, standardFullClearRenderPassDescriptor } from "../gfx/helpers/RenderGraphHelpers.js";
import { fillVec4 } from "../gfx/helpers/UniformBufferHelpers.js";
import { GfxDevice, GfxFormat, GfxMipFilterMode, GfxSampler, GfxTexFilterMode, GfxWrapMode } from "../gfx/platform/GfxPlatform.js";
import { GfxrAttachmentSlot, GfxrGraphBuilder, GfxrRenderTargetDescription, GfxrRenderTargetID } from "../gfx/render/GfxRenderGraph.js";
import { GfxRenderHelper } from "../gfx/render/GfxRenderHelper.js";
import { GfxRenderInst } from "../gfx/render/GfxRenderInstManager.js";
import { DeviceProgram } from "../Program.js";
import { TextureMapping } from "../TextureHolder.js";
import { ViewerRenderInput } from "../viewer.js";
import { TheWitnessGlobals } from "./Globals.js";

// Average the log of the luminance, so the mean is geometric: a few bright pixels shouldn't
// decide the exposure for the whole frame.
class Luminance_Program extends DeviceProgram {
    public override vert = GfxShaderLibrary.fullscreenVS;

    public override frag = `
uniform sampler2D u_Texture;
in vec2 v_TexCoord;

void main() {
    vec2 t_Step = 1.0 / vec2(textureSize(TEXTURE(u_Texture), 0)) * u_Taps;
    float t_Total = 0.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            vec2 t_Offset = (vec2(float(x), float(y)) - 1.5) * t_Step;
            vec3 t_Color = texture(SAMPLER_2D(u_Texture), v_TexCoord + t_Offset).rgb;
            t_Total += log(max(dot(t_Color, vec3(0.2126, 0.7152, 0.0722)), 0.0001));
        }
    }
    gl_FragColor = vec4(t_Total / 16.0, 0.0, 0.0, 1.0);
}
`;

    constructor(private first: boolean) {
        super();
        // The first pass reads the scene and takes the log; later ones average what's already
        // logarithmic, so they read it straight.
        if (!first) {
            this.frag = this.frag.replace(`log(max(dot(t_Color, vec3(0.2126, 0.7152, 0.0722)), 0.0001))`, `t_Color.r`);
        }
        this.frag = this.frag.replace(`u_Taps`, `2.0`);
    }
}

// Everything above the threshold, at quarter resolution, ready to be blurred into a glow.
class Bright_Pass_Program extends DeviceProgram {
    public override vert = GfxShaderLibrary.fullscreenVS;

    public override frag = `
uniform sampler2D u_Texture;
uniform sampler2D u_Luminance;
in vec2 v_TexCoord;

layout(std140) uniform ub_Params {
    vec4 u_Params;
};

#define u_Threshold     (u_Params.x)
#define u_KeyValue      (u_Params.y)
#define u_MinLuminance  (u_Params.z)
#define u_MaxLuminance  (u_Params.w)

// The scene's units are not the game's: a lit outdoor frame meters about 2.5 here, against the
// band the game states for its own luminance. This puts the two on the same footing so the
// bounds can be the game's own min_luminance and max_luminance. Bounding the luminance rather
// than the exposure is the point of them -- it holds the exposure inside a factor of ten, where
// bounding the exposure directly let it swing thirtyfold between a lit garden and a cave, and a
// threshold measured after exposure then meant something different in every room.
#define SCENE_LUMINANCE_SCALE (0.4)

float CalcExposure(in float t_AverageLuminance, in float t_KeyValue, in float t_MinLuminance, in float t_MaxLuminance) {
    float t_Luminance = clamp(t_AverageLuminance * SCENE_LUMINANCE_SCALE, t_MinLuminance, t_MaxLuminance);
    return t_KeyValue / t_Luminance;
}

void main() {
    // The same exposure the composite will apply, so the threshold means the same thing in both.
    float t_AverageLuminance = exp(texture(SAMPLER_2D(u_Luminance), vec2(0.5)).r);
    float t_Exposure = CalcExposure(t_AverageLuminance, u_KeyValue, u_MinLuminance, u_MaxLuminance);
    vec3 t_Color = texture(SAMPLER_2D(u_Texture), v_TexCoord).rgb * t_Exposure;

    // How far past the threshold the pixel is, measured on brightness rather than per channel:
    // thresholding each channel on its own makes a saturated colour glow for being red, not for
    // being bright, which is what turned the blossom trees into lamps.
    float t_Luma = dot(t_Color, vec3(0.2126, 0.7152, 0.0722));
    gl_FragColor = vec4(t_Color * (max(t_Luma - u_Threshold, 0.0) / max(t_Luma, 0.0001)), 1.0);
}
`;
}

class Blur_Program extends DeviceProgram {
    public override vert = GfxShaderLibrary.fullscreenVS;

    public override frag = `
uniform sampler2D u_Texture;
in vec2 v_TexCoord;

void main() {
    vec2 t_Size = vec2(textureSize(TEXTURE(u_Texture), 0));
#ifdef HORIZONTAL
    vec2 t_Step = vec2(1.0, 0.0) / t_Size;
#else
    vec2 t_Step = vec2(0.0, 1.0) / t_Size;
#endif

    // A gaussian of sigma 2.18, the width the game asks for, folded onto bilinear taps.
    vec3 t_Color = texture(SAMPLER_2D(u_Texture), v_TexCoord).rgb * 0.1964;
    t_Color += texture(SAMPLER_2D(u_Texture), v_TexCoord + t_Step * 1.4104).rgb * 0.2969;
    t_Color += texture(SAMPLER_2D(u_Texture), v_TexCoord - t_Step * 1.4104).rgb * 0.2969;
    t_Color += texture(SAMPLER_2D(u_Texture), v_TexCoord + t_Step * 3.2979).rgb * 0.1049;
    t_Color += texture(SAMPLER_2D(u_Texture), v_TexCoord - t_Step * 3.2979).rgb * 0.1049;
    gl_FragColor = vec4(t_Color, 1.0);
}
`;
}

// Exposure, bloom, the filmic curve and the vignette, in that order, on the way to the screen.
class Composite_Program extends DeviceProgram {
    public override vert = GfxShaderLibrary.fullscreenVS;

    public override frag = `
uniform sampler2D u_Texture;
uniform sampler2D u_Bloom;
uniform sampler2D u_Luminance;
in vec2 v_TexCoord;

layout(std140) uniform ub_Params {
    vec4 u_Params0;
    vec4 u_Params1;
};

#define u_KeyValue      (u_Params0.x)
#define u_MinLuminance  (u_Params0.y)
#define u_MaxLuminance  (u_Params0.z)
#define u_BloomWeight   (u_Params0.w)
#define u_Vignetting    (u_Params1.x)
#define u_BloomEnabled  (u_Params1.y)

// The scene's units are not the game's: a lit outdoor frame meters about 2.5 here, against the
// band the game states for its own luminance. This puts the two on the same footing so the
// bounds can be the game's own min_luminance and max_luminance. Bounding the luminance rather
// than the exposure is the point of them -- it holds the exposure inside a factor of ten, where
// bounding the exposure directly let it swing thirtyfold between a lit garden and a cave, and a
// threshold measured after exposure then meant something different in every room.
#define SCENE_LUMINANCE_SCALE (0.4)

float CalcExposure(in float t_AverageLuminance, in float t_KeyValue, in float t_MinLuminance, in float t_MaxLuminance) {
    float t_Luminance = clamp(t_AverageLuminance * SCENE_LUMINANCE_SCALE, t_MinLuminance, t_MaxLuminance);
    return t_KeyValue / t_Luminance;
}

float Uncharted2Tonemap(float x) {
    float A = 0.15, B = 0.5, C = 0.1, D = 0.1, E = 0.02, F = 0.6;
    return (((x * ((A * x) + (C * B))) + (D * E)) / ((x * ((A * x) + B)) + (D * F))) - (E / F);
}

void main() {
    vec3 t_Color = texture(SAMPLER_2D(u_Texture), v_TexCoord).rgb;

    // The frame's own brightness sets the exposure, the way the game's auto-exposure does: the
    // key value over the geometric mean of the luminance, so the average lands on middle grey.
    float t_AverageLuminance = exp(texture(SAMPLER_2D(u_Luminance), vec2(0.5)).r);
    float t_Exposure = CalcExposure(t_AverageLuminance, u_KeyValue, u_MinLuminance, u_MaxLuminance);
    t_Color *= t_Exposure;

    if (u_BloomEnabled > 0.0)
        t_Color += texture(SAMPLER_2D(u_Bloom), v_TexCoord).rgb * u_BloomWeight;

    // Filmic curve, on luminance, so hue survives it.
    float t_Luma = max(max(max(t_Color.x, t_Color.y), t_Color.z), 0.01);
    float W = 32.0;
    float t_Tonemapped = Uncharted2Tonemap(t_Luma) * (1.0 / Uncharted2Tonemap(W));
    t_Color *= t_Tonemapped / t_Luma;

    // Vignette: a gentle darkening towards the corners.
    vec2 t_FromCenter = v_TexCoord - 0.5;
    float t_Vignette = 1.0 - u_Vignetting * dot(t_FromCenter, t_FromCenter) * 0.85;
    t_Color *= t_Vignette;

    gl_FragColor = vec4(pow(t_Color, vec3(1.0 / 2.2)), 1.0);
}
`;
}

export class Post_Process {
    private luminanceFirstProgram = new Luminance_Program(true);
    private luminanceProgram = new Luminance_Program(false);
    private brightPassProgram = new Bright_Pass_Program();
    private blurXProgram: Blur_Program;
    private blurYProgram: Blur_Program;
    private compositeProgram = new Composite_Program();
    private sampler: GfxSampler;
    private textureMapping = [new TextureMapping(), new TextureMapping(), new TextureMapping()];

    constructor(device: GfxDevice, renderHelper: GfxRenderHelper) {
        this.blurXProgram = new Blur_Program();
        this.blurXProgram.defines.set('HORIZONTAL', '1');
        this.blurYProgram = new Blur_Program();

        this.sampler = renderHelper.renderCache.createSampler({
            minFilter: GfxTexFilterMode.Bilinear,
            magFilter: GfxTexFilterMode.Bilinear,
            mipFilter: GfxMipFilterMode.Nearest,
            wrapS: GfxWrapMode.Clamp,
            wrapT: GfxWrapMode.Clamp,
        });
        for (const mapping of this.textureMapping)
            mapping.gfxSampler = this.sampler;
    }

    private makeDesc(width: number, height: number, format: GfxFormat): GfxrRenderTargetDescription {
        const desc = new GfxrRenderTargetDescription(format);
        desc.setDimensions(Math.max(width, 1), Math.max(height, 1), 1);
        desc.clearColor = standardFullClearRenderPassDescriptor.clearColor;
        return desc;
    }

    public render(globals: TheWitnessGlobals, builder: GfxrGraphBuilder, renderHelper: GfxRenderHelper, viewerInput: ViewerRenderInput, mainColorTargetID: GfxrRenderTargetID): GfxrRenderTargetID {
        const renderInstManager = renderHelper.renderInstManager;
        const cache = renderInstManager.gfxRenderCache;

        const tone_mapping = globals.all_variables['render/tone_mapping'];
        const bloom = globals.all_variables['render/bloom'];
        // All of these come out of tone_mapping.variables and bloom.variables, which Globals now
        // loads alongside All.variables. The fallbacks are those same files' values, for a
        // universe that ships without them.
        const key_value = tone_mapping !== undefined ? tone_mapping.key_value as number : 0.22;
        const min_luminance = tone_mapping !== undefined ? tone_mapping.min_luminance as number : 0.1;
        const max_luminance = tone_mapping !== undefined ? tone_mapping.max_luminance as number : 1.0;
        const vignetting = tone_mapping !== undefined ? tone_mapping.vignetting as number : 0.801653;
        const bloom_enabled = bloom !== undefined ? bloom.enable_bloom !== false : true;
        const bloom_threshold = bloom !== undefined ? bloom.bloom_threshold as number : 10.0;
        const bloom_weight = bloom !== undefined ? bloom.bloom_weight as number : 0.743802;

        const fullscreen = (): GfxRenderInst => {
            const renderInst = renderInstManager.newRenderInst();
            renderInst.setUniformBuffer(renderHelper.uniformBuffer);
            renderInst.setMegaStateFlags(fullscreenMegaState);
            renderInst.setVertexInput(null, null, null);
            renderInst.setDrawCount(3);
            return renderInst;
        };

        // Work the frame's luminance down to a single value, a quarter of the way each pass.
        let luminanceTargetID: GfxrRenderTargetID | null = null;
        const luminanceSizes = [64, 8, 1];
        for (let i = 0; i < luminanceSizes.length; i++) {
            const size = luminanceSizes[i];
            const targetID = builder.createRenderTargetID(this.makeDesc(size, size, GfxFormat.F16_RGBA), `Luminance ${size}`);
            const sourceID = luminanceTargetID !== null ? luminanceTargetID : mainColorTargetID;
            const program = i === 0 ? this.luminanceFirstProgram : this.luminanceProgram;

            const renderInst = fullscreen();
            renderInst.setBindingLayouts([{ numUniformBuffers: 0, numSamplers: 1 }]);
            renderInst.setGfxProgram(cache.createProgram(program));

            builder.pushPass((pass) => {
                pass.setDebugName(`Luminance ${size}`);
                pass.attachRenderTargetID(GfxrAttachmentSlot.Color0, targetID);
                const resolveID = builder.resolveRenderTarget(sourceID);
                pass.attachResolveTexture(resolveID);
                pass.exec((passRenderer, scope) => {
                    this.textureMapping[0].gfxTexture = scope.getResolveTextureForID(resolveID);
                    renderInst.setSamplerBindingsFromTextureMappings([this.textureMapping[0]]);
                    renderInst.drawOnPass(cache, passRenderer);
                });
            });

            luminanceTargetID = targetID;
        }

        // The glow: everything past the threshold, blurred at quarter resolution.
        let bloomTargetID: GfxrRenderTargetID | null = null;
        if (bloom_enabled) {
            const bloomDesc = this.makeDesc(viewerInput.backbufferWidth >> 2, viewerInput.backbufferHeight >> 2, GfxFormat.F16_RGBA);
            const brightTargetID = builder.createRenderTargetID(bloomDesc, 'Bloom Bright Pass');

            const brightInst = fullscreen();
            brightInst.setBindingLayouts([{ numUniformBuffers: 1, numSamplers: 2 }]);
            brightInst.setGfxProgram(cache.createProgram(this.brightPassProgram));
            {
                const offs = brightInst.allocateUniformBuffer(0, 4);
                fillVec4(brightInst.mapUniformBufferF32(0), offs, bloom_threshold, key_value, min_luminance, max_luminance);
            }

            builder.pushPass((pass) => {
                pass.setDebugName('Bloom Bright Pass');
                pass.attachRenderTargetID(GfxrAttachmentSlot.Color0, brightTargetID);
                const colorResolveID = builder.resolveRenderTarget(mainColorTargetID);
                pass.attachResolveTexture(colorResolveID);
                const luminanceResolveID = builder.resolveRenderTarget(luminanceTargetID!);
                pass.attachResolveTexture(luminanceResolveID);
                pass.exec((passRenderer, scope) => {
                    this.textureMapping[0].gfxTexture = scope.getResolveTextureForID(colorResolveID);
                    this.textureMapping[1].gfxTexture = scope.getResolveTextureForID(luminanceResolveID);
                    brightInst.setSamplerBindingsFromTextureMappings([this.textureMapping[0], this.textureMapping[1]]);
                    brightInst.drawOnPass(cache, passRenderer);
                });
            });

            let blurSourceID = brightTargetID;
            for (const horizontal of [true, false, true, false]) {
                const blurTargetID = builder.createRenderTargetID(bloomDesc, horizontal ? 'Bloom Blur X' : 'Bloom Blur Y');
                const sourceID = blurSourceID;
                const blurInst = fullscreen();
                blurInst.setBindingLayouts([{ numUniformBuffers: 0, numSamplers: 1 }]);
                blurInst.setGfxProgram(cache.createProgram(horizontal ? this.blurXProgram : this.blurYProgram));

                builder.pushPass((pass) => {
                    pass.setDebugName(horizontal ? 'Bloom Blur X' : 'Bloom Blur Y');
                    pass.attachRenderTargetID(GfxrAttachmentSlot.Color0, blurTargetID);
                    const resolveID = builder.resolveRenderTarget(sourceID);
                    pass.attachResolveTexture(resolveID);
                    pass.exec((passRenderer, scope) => {
                        this.textureMapping[0].gfxTexture = scope.getResolveTextureForID(resolveID);
                        blurInst.setSamplerBindingsFromTextureMappings([this.textureMapping[0]]);
                        blurInst.drawOnPass(cache, passRenderer);
                    });
                });
                blurSourceID = blurTargetID;
            }

            bloomTargetID = blurSourceID;
        }

        // And out to the screen.
        const ldrTargetID = builder.createRenderTargetID(makeBackbufferDescSimple(GfxrAttachmentSlot.Color0, viewerInput, standardFullClearRenderPassDescriptor), 'Tone Mapped');

        const compositeInst = fullscreen();
        compositeInst.setBindingLayouts([{ numUniformBuffers: 1, numSamplers: 3 }]);
        compositeInst.setGfxProgram(cache.createProgram(this.compositeProgram));
        {
            let offs = compositeInst.allocateUniformBuffer(0, 8);
            const d = compositeInst.mapUniformBufferF32(0);
            offs += fillVec4(d, offs, key_value, min_luminance, max_luminance, bloom_weight);
            offs += fillVec4(d, offs, vignetting, bloomTargetID !== null ? 1 : 0, 0, 0);
        }

        builder.pushPass((pass) => {
            pass.setDebugName('Tone Map');
            pass.attachRenderTargetID(GfxrAttachmentSlot.Color0, ldrTargetID);

            const colorResolveID = builder.resolveRenderTarget(mainColorTargetID);
            pass.attachResolveTexture(colorResolveID);
            const luminanceResolveID = builder.resolveRenderTarget(luminanceTargetID!);
            pass.attachResolveTexture(luminanceResolveID);
            const bloomResolveID = bloomTargetID !== null ? builder.resolveRenderTarget(bloomTargetID) : null;
            if (bloomResolveID !== null)
                pass.attachResolveTexture(bloomResolveID);

            pass.exec((passRenderer, scope) => {
                this.textureMapping[0].gfxTexture = scope.getResolveTextureForID(colorResolveID);
                this.textureMapping[1].gfxTexture = bloomResolveID !== null ? scope.getResolveTextureForID(bloomResolveID) : scope.getResolveTextureForID(colorResolveID);
                this.textureMapping[2].gfxTexture = scope.getResolveTextureForID(luminanceResolveID);
                compositeInst.setSamplerBindingsFromTextureMappings(this.textureMapping);
                compositeInst.drawOnPass(cache, passRenderer);
            });
        });

        return ldrTargetID;
    }
}

