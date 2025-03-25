"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";
import * as THREE from "three";
import { flushSync } from "react-dom"; // For NumericInput commit logic

// Shadcn UI Components
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "sonner"; // Assuming sonner for toast notifications
import { extend } from "@react-three/fiber";

// Tailwind utility
import { cn } from "@/lib/utils";

// --- Ported Shaders ---
const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  precision mediump float;
  varying vec2 vUv;

  uniform sampler2D u_image_texture;
  uniform float u_time;
  uniform float u_ratio; // canvas aspect ratio (width / height)
  uniform float u_img_ratio; // image aspect ratio (width / height)
  uniform float u_patternScale;
  uniform float u_refraction;
  uniform float u_edge;
  uniform float u_patternBlur;
  uniform float u_liquid;
  uniform vec2 u_resolution; // canvas resolution

  // --- snoise function (copied directly from original) ---
  vec3 mod289(vec3 x) { return x - floor(x * (1. / 289.)) * 289.; }
  vec2 mod289(vec2 x) { return x - floor(x * (1. / 289.)) * 289.; }
  vec3 permute(vec3 x) { return mod289(((x*34.)+1.)*x); }
  float snoise(vec2 v) {
      const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
      vec2 i = floor(v + dot(v, C.yy));
      vec2 x0 = v - i + dot(i, C.xx);
      vec2 i1;
      i1 = (x0.x > x0.y) ? vec2(1., 0.) : vec2(0., 1.);
      vec4 x12 = x0.xyxy + C.xxzz;
      x12.xy -= i1;
      i = mod289(i);
      vec3 p = permute(permute(i.y + vec3(0., i1.y, 1.)) + i.x + vec3(0., i1.x, 1.));
      vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.);
      m = m*m;
      m = m*m;
      vec3 x = 2. * fract(p * C.www) - 1.;
      vec3 h = abs(x) - 0.5;
      vec3 ox = floor(x + 0.5);
      vec3 a0 = x - ox;
      m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
      vec3 g;
      g.x = a0.x * x0.x + h.x * x0.y;
      g.yz = a0.yz * x12.xz + h.yz * x12.yw;
      return 130. * dot(m, g);
  }
  // --- End snoise ---

  #define TWO_PI 6.28318530718
  #define PI 3.14159265358979323846

  vec2 get_img_uv() {
      // We map the shader to a plane filling the screen, vUv goes from 0 to 1
      vec2 uv = vUv;
      float canvasAspect = u_resolution.x / u_resolution.y;
      float imageAspect = u_img_ratio;
      vec2 scale = vec2(1.0);

      if (canvasAspect > imageAspect) {
          // Canvas wider than image
          scale.x = canvasAspect / imageAspect;
      } else {
          // Canvas taller than image
          scale.y = imageAspect / canvasAspect;
      }

      uv = (uv - 0.5) * scale + 0.5;

      // Match original Y direction if necessary (depends on texture loading)
      // uv.y = 1.0 - uv.y; // R3F/Drei usually handle this

      return uv;
  }

  vec2 rotate(vec2 uv, float th) {
    return mat2(cos(th), sin(th), -sin(th), cos(th)) * uv;
  }

  float get_color_channel(float c1, float c2, float stripe_p, vec3 w, float extra_blur, float b) {
      float ch = c2;
      float border = 0.;
      float blur = u_patternBlur + extra_blur;

      ch = mix(ch, c1, smoothstep(.0, blur, stripe_p));

      border = w[0];
      ch = mix(ch, c2, smoothstep(border - blur, border + blur, stripe_p));

      b = smoothstep(.2, .8, b);
      border = w[0] + .4 * (1. - b) * w[1];
      ch = mix(ch, c1, smoothstep(border - blur, border + blur, stripe_p));

      border = w[0] + .5 * (1. - b) * w[1];
      ch = mix(ch, c2, smoothstep(border - blur, border + blur, stripe_p));

      border = w[0] + w[1];
      ch = mix(ch, c1, smoothstep(border - blur, border + blur, stripe_p));

      float gradient_t = (stripe_p - w[0] - w[1]) / w[2];
      float gradient = mix(c1, c2, smoothstep(0., 1., gradient_t));
      ch = mix(ch, gradient, smoothstep(border - blur, border + blur, stripe_p));

      return ch;
  }

  float get_img_frame_alpha(vec2 uv, float img_frame_width) {
      // Check if uv is within the 0-1 range considering the aspect ratio correction
      float alpha_x = smoothstep(0.0, img_frame_width, uv.x) * smoothstep(1.0, 1.0 - img_frame_width, uv.x);
      float alpha_y = smoothstep(0.0, img_frame_width, uv.y) * smoothstep(1.0, 1.0 - img_frame_width, uv.y);
      return alpha_x * alpha_y;
  }


  void main() {
      vec2 uv = vUv; // Use direct vUv for screen space calculations
      // uv.y = 1. - uv.y; // Flip Y if needed for calculations

      float canvasAspect = u_resolution.x / u_resolution.y;
      uv.x *= canvasAspect; // Correct aspect ratio for screen space calcs


      float diagonal = uv.x - uv.y;
      float t = .001 * u_time;

      vec2 img_uv = get_img_uv(); // UV for texture sampling
      vec4 img = texture2D(u_image_texture, img_uv);

      vec3 color = vec3(0.);
      float opacity = 1.;

      vec3 color1 = vec3(.98, 0.98, 1.);
      vec3 color2 = vec3(.1, .1, .1 + .1 * smoothstep(.7, 1.3, uv.x + uv.y)); // Use screen space uv

      float edge = img.r; // Assuming processed image stores edge info in red channel

      // --- Calculations mostly copied from original ---
      vec2 grad_uv = uv;
      grad_uv -= 0.5 * vec2(canvasAspect, 1.0); // Center screen UV considering aspect

      float dist = length(grad_uv + vec2(0., .2 * diagonal));

      grad_uv = rotate(grad_uv, (.25 - .2 * diagonal) * PI);

      float bulge = pow(1.8 * dist, 1.2);
      bulge = 1. - bulge;
      bulge *= pow(vUv.y, .3); // Use original vUv.y for this vertical effect

      float cycle_width = u_patternScale;
      float thin_strip_1_ratio = .12 / cycle_width * (1. - .4 * bulge);
      float thin_strip_2_ratio = .07 / cycle_width * (1. + .4 * bulge);
      float wide_strip_ratio = (1. - thin_strip_1_ratio - thin_strip_2_ratio);

      float thin_strip_1_width = cycle_width * thin_strip_1_ratio;
      float thin_strip_2_width = cycle_width * thin_strip_2_ratio;

      opacity = 1. - smoothstep(.9 - .5 * u_edge, 1. - .5 * u_edge, edge);
      opacity *= get_img_frame_alpha(img_uv, 0.01); // Use image UVs for frame


      float noise = snoise(uv - t); // Use screen space uv for noise

      edge += (1. - edge) * u_liquid * noise;

      float refr = 0.;
      refr += (1. - bulge);
      refr = clamp(refr, 0., 1.);

      float dir = grad_uv.x;

      dir += diagonal;
      dir -= 2. * noise * diagonal * (smoothstep(0., 1., edge) * smoothstep(1., 0., edge));

      bulge *= clamp(pow(vUv.y, .1), .3, 1.); // Use original vUv.y
      dir *= (.1 + (1.1 - edge) * bulge);
      dir *= smoothstep(1., .7, edge);

      dir += .18 * (smoothstep(.1, .2, vUv.y) * smoothstep(.4, .2, vUv.y)); // Use original vUv.y
      dir += .03 * (smoothstep(.1, .2, 1. - vUv.y) * smoothstep(.4, .2, 1. - vUv.y)); // Use original vUv.y

      dir *= (.5 + .5 * pow(vUv.y, 2.)); // Use original vUv.y

      dir *= cycle_width;
      dir -= t;

      float refr_r = refr;
      refr_r += .03 * bulge * noise;
      float refr_b = 1.3 * refr;

      refr_r += 5. * (smoothstep(-.1, .2, vUv.y) * smoothstep(.5, .1, vUv.y)) * (smoothstep(.4, .6, bulge) * smoothstep(1., .4, bulge)); // Use original vUv.y
      refr_r -= diagonal;

      refr_b += (smoothstep(0., .4, vUv.y) * smoothstep(.8, .1, vUv.y)) * (smoothstep(.4, .6, bulge) * smoothstep(.8, .4, bulge)); // Use original vUv.y
      refr_b -= .2 * edge;

      refr_r *= u_refraction;
      refr_b *= u_refraction;

      vec3 w = vec3(thin_strip_1_width, thin_strip_2_width, wide_strip_ratio);
      w[1] -= .02 * smoothstep(.0, 1., edge + bulge);
      float stripe_r = mod(dir + refr_r, 1.);
      float r = get_color_channel(color1.r, color2.r, stripe_r, w, 0.02 + .03 * u_refraction * bulge, bulge);
      float stripe_g = mod(dir, 1.);
      float g = get_color_channel(color1.g, color2.g, stripe_g, w, 0.01 / (1. - diagonal), bulge);
      float stripe_b = mod(dir - refr_b, 1.);
      float b = get_color_channel(color1.b, color2.b, stripe_b, w, .01, bulge);

      color = vec3(r, g, b);
      color *= opacity;

      // If the texture lookup is outside the valid range (0-1), make it transparent
      if (img_uv.x < 0.0 || img_uv.x > 1.0 || img_uv.y < 0.0 || img_uv.y > 1.0) {
         opacity = 0.0;
      }


      gl_FragColor = vec4(color, opacity);
  }
`;

// --- R3F Shader Material ---
const LiquidShaderMaterial = shaderMaterial(
  // Uniforms
  {
    u_time: 0,
    u_image_texture: null,
    u_resolution: new THREE.Vector2(),
    u_ratio: 1, // canvas aspect ratio
    u_img_ratio: 1, // image aspect ratio
    u_refraction: 0.015,
    u_edge: 0.4,
    u_patternBlur: 0.005,
    u_liquid: 0.07,
    u_speed: 0.3,
    u_patternScale: 2,
  },
  // Vertex Shader
  vertexShader,
  // Fragment Shader
  fragmentShader
);

extend({ LiquidShaderMaterial });

// Extend THREE namespace for JSX shaderMaterial
declare global {
  namespace JSX {
    interface IntrinsicElements {
      liquidShaderMaterial: any; // Adjust type if possible
    }
  }
}

// --- Image Processing Logic (Ported) ---
// This function needs careful porting and testing. It's complex.
async function processImageForShader(
  file: File
): Promise<{ imageData: ImageData; pngBlob: Blob } | null> {
  try {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);

    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = objectUrl;
    });
    URL.revokeObjectURL(objectUrl);

    let targetWidth = image.naturalWidth;
    let targetHeight = image.naturalHeight;

    if (file.type === "image/svg+xml") {
      // Handle SVG size
      targetWidth = 1000;
      targetHeight = 1000;
    } else {
      // Resize non-SVG images
      const maxSize = 1000;
      const minSize = 500;
      if (
        targetWidth > maxSize ||
        targetHeight > maxSize ||
        targetWidth < minSize ||
        targetHeight < minSize
      ) {
        if (targetWidth > targetHeight) {
          if (targetWidth > maxSize) {
            targetHeight = Math.round((maxSize * targetHeight) / targetWidth);
            targetWidth = maxSize;
          } else if (targetWidth < minSize) {
            targetHeight = Math.round((minSize * targetHeight) / targetWidth);
            targetWidth = minSize;
          }
        } else {
          if (targetHeight > maxSize) {
            targetWidth = Math.round((maxSize * targetWidth) / targetHeight);
            targetHeight = maxSize;
          } else if (targetHeight < minSize) {
            targetWidth = Math.round((minSize * targetWidth) / targetHeight);
            targetHeight = minSize;
          }
        }
      }
    }

    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not get 2D context");

    ctx.drawImage(image, 0, 0, targetWidth, targetHeight);
    const originalImageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const data = originalImageData.data;

    // 1. Create occupancy map (f)
    const isOccupied = new Array(targetWidth * targetHeight).fill(false);
    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const i = (y * targetWidth + x) * 4;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const a = data[i + 3];
        // Check if not white/transparent
        if (!(r === 255 && g === 255 && b === 255 && a === 255) && a !== 0) {
          isOccupied[y * targetWidth + x] = true;
        }
      }
    }

    // 2. Create edge map (x)
    const isEdge = new Array(targetWidth * targetHeight).fill(false);
    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const idx = y * targetWidth + x;
        if (isOccupied[idx]) {
          let foundEmptyNeighbor = false;
          for (let ny = y - 1; ny <= y + 1; ny++) {
            for (let nx = x - 1; nx <= x + 1; nx++) {
              if (nx >= 0 && nx < targetWidth && ny >= 0 && ny < targetHeight) {
                if (!isOccupied[ny * targetWidth + nx]) {
                  foundEmptyNeighbor = true;
                  break;
                }
              } else {
                // Pixel is on the border of the image and is occupied
                foundEmptyNeighbor = true;
                break;
              }
            }
            if (foundEmptyNeighbor) break;
          }
          if (foundEmptyNeighbor) {
            isEdge[idx] = true;
          }
        }
      }
    }

    // 3. Calculate distance field (T) using iterative approach
    let distanceField = new Float32Array(targetWidth * targetHeight).fill(0);
    let tempDistanceField = new Float32Array(targetWidth * targetHeight).fill(
      0
    );
    const iterations = 300; // As per original code

    const getDistance = (x: number, y: number, field: Float32Array): number => {
      if (
        x < 0 ||
        x >= targetWidth ||
        y < 0 ||
        y >= targetHeight ||
        !isOccupied[y * targetWidth + x]
      ) {
        return 0;
      }
      return field[y * targetWidth + x];
    };

    for (let iter = 0; iter < iterations; iter++) {
      for (let y = 0; y < targetHeight; y++) {
        for (let x = 0; x < targetWidth; x++) {
          const idx = y * targetWidth + x;
          if (!isOccupied[idx] || isEdge[idx]) {
            tempDistanceField[idx] = 0;
            continue;
          }
          const neighborsSum =
            getDistance(x + 1, y, distanceField) +
            getDistance(x - 1, y, distanceField) +
            getDistance(x, y + 1, distanceField) +
            getDistance(x, y - 1, distanceField);
          tempDistanceField[idx] = (0.01 + neighborsSum) / 4.0;
        }
      }
      // Swap buffers
      [distanceField, tempDistanceField] = [tempDistanceField, distanceField];
    }

    // 4. Normalize distance field and create final ImageData
    let maxDistance = 0;
    for (let i = 0; i < distanceField.length; i++) {
      if (distanceField[i] > maxDistance) {
        maxDistance = distanceField[i];
      }
    }

    const finalImageData = ctx.createImageData(targetWidth, targetHeight);
    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const idx = y * targetWidth + x;
        const i = idx * 4;
        if (isOccupied[idx]) {
          // Use distance field for grayscale value (inverted and squared like original)
          const normalizedDistance =
            maxDistance > 0 ? distanceField[idx] / maxDistance : 0;
          const value = 255 * (1 - Math.pow(normalizedDistance, 2));
          finalImageData.data[i] = value; // R
          finalImageData.data[i + 1] = value; // G
          finalImageData.data[i + 2] = value; // B
          finalImageData.data[i + 3] = 255; // A
        } else {
          // Background (white)
          finalImageData.data[i] = 255;
          finalImageData.data[i + 1] = 255;
          finalImageData.data[i + 2] = 255;
          finalImageData.data[i + 3] = 255;
        }
      }
    }

    // 5. Create PNG Blob
    ctx.putImageData(finalImageData, 0, 0); // Put processed data back onto canvas
    const pngBlob = await new Promise<Blob | null>(resolve => {
      canvas.toBlob(resolve, "image/png");
    });

    if (!pngBlob) throw new Error("Failed to create PNG blob");

    return { imageData: finalImageData, pngBlob };
  } catch (error) {
    console.error("Error processing image:", error);
    toast.error("Failed to process image.");
    return null;
  }
}

// --- Helper: Debounce Function ---
function debounce<T extends (...args: any[]) => void>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, wait);
  };
}

// --- Helper: Round Number ---
// From original: v = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
const roundFactors = [1, 10, 100, 1000, 10000, 100000, 1000000];
function roundToPrecision(value: number, precision: number = 2): number {
  const factor = roundFactors[precision] ?? Math.pow(10, precision);
  const temp = value * factor;
  const roundedTemp = Math.trunc(temp + (temp > 0 ? 0.5 : -0.5));
  const result = roundedTemp / factor;
  return result === 0 ? 0 : result; // Handle -0 case
}

// --- Default Params ---
const paramConfig = {
  refraction: { min: 0, max: 0.06, step: 0.001, default: 0.015 },
  edge: { min: 0, max: 1, step: 0.01, default: 0.4 },
  patternBlur: { min: 0, max: 0.05, step: 0.001, default: 0.005 },
  liquid: { min: 0, max: 1, step: 0.01, default: 0.07 },
  speed: { min: 0, max: 1, step: 0.01, default: 0.3 },
  patternScale: { min: 1, max: 10, step: 0.1, default: 2 },
};

const defaultParams = Object.fromEntries(
  Object.entries(paramConfig).map(([key, config]) => [key, config.default])
) as Record<keyof typeof paramConfig, number> & { background: string };
defaultParams.background = "metal"; // Add background default

// --- Shader Component ---
interface ShaderEffectProps {
  imageData: ImageData | null;
  params: typeof defaultParams;
}

function ShaderEffect({ imageData, params }: ShaderEffectProps) {
  const shaderRef = useRef<THREE.ShaderMaterial>(null);
  const texture = useMemo(() => {
    return imageData ? new THREE.CanvasTexture(imageData as any) : null; // Use CanvasTexture for ImageData
  }, [imageData]);

  const { size } = useThree(); // Get canvas size

  useEffect(() => {
    if (texture) {
      texture.needsUpdate = true; // Ensure texture updates
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
    }
    if (shaderRef.current) {
      shaderRef.current.uniforms.u_image_texture.value = texture;
      shaderRef.current.uniforms.u_img_ratio.value = texture
        ? texture.image.width / texture.image.height
        : 1;
    }
  }, [texture]);

  useEffect(() => {
    if (shaderRef.current) {
      shaderRef.current.uniforms.u_resolution.value.set(
        size.width,
        size.height
      );
      shaderRef.current.uniforms.u_ratio.value = size.width / size.height;
    }
  }, [size]);

  useFrame(({ clock }) => {
    if (shaderRef.current) {
      // Update uniforms based on params state
      shaderRef.current.uniforms.u_time.value =
        clock.elapsedTime * 1000 * params.speed; // Match original time scale
      shaderRef.current.uniforms.u_refraction.value = params.refraction;
      shaderRef.current.uniforms.u_edge.value = params.edge;
      shaderRef.current.uniforms.u_patternBlur.value = params.patternBlur;
      shaderRef.current.uniforms.u_liquid.value = params.liquid;
      shaderRef.current.uniforms.u_patternScale.value = params.patternScale;
      // u_speed is handled by scaling time
    }
  });

  if (!imageData) return null; // Don't render plane if no image

  return (
    <mesh>
      {/* Full screen quad */}
      <planeGeometry args={[2, 2]} />
      <liquidShaderMaterial
        ref={shaderRef}
        key={LiquidShaderMaterial.key} // Needed for HMR
        transparent={true}
        blending={THREE.NormalBlending}
      />
    </mesh>
  );
}

// --- UI Component for a single Parameter Slider + Input ---
interface ParamControlProps {
  label: string;
  paramKey: keyof typeof defaultParams;
  params: typeof defaultParams;
  config: { min: number; max: number; step: number; default: number };
  onParamChange: (key: keyof typeof defaultParams, value: number) => void;
  formatValue?: (value: number) => string;
}

function ParamControl({
  label,
  paramKey,
  params,
  config,
  onParamChange,
  formatValue,
}: ParamControlProps) {
  const value = params[paramKey] as number;
  const formattedValue = formatValue ? formatValue(value) : value.toFixed(3); // Default format

  const handleSliderChange = (newValue: number[]) => {
    onParamChange(paramKey, newValue[0]);
  };

  // State for the input field to allow intermediate typing
  const [inputValue, setInputValue] = useState(formattedValue);
  const inputRef = useRef<HTMLInputElement>(null);
  const isTyping = useRef(false);

  // Update input when slider changes (and not currently typing)
  useEffect(() => {
    if (!isTyping.current) {
      setInputValue(formattedValue);
    }
  }, [formattedValue]);

  const commitValue = (currentValStr: string) => {
    let numValue = parseFloat(currentValStr);
    if (!isNaN(numValue)) {
      numValue = Math.max(config.min, Math.min(config.max, numValue)); // Clamp
      onParamChange(paramKey, numValue); // Commit clamped value
      setInputValue(formatValue ? formatValue(numValue) : numValue.toFixed(3)); // Update input to formatted committed value
    } else {
      setInputValue(formattedValue); // Revert if invalid
    }
    isTyping.current = false;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    isTyping.current = true;
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    commitValue(inputValue);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      commitValue(inputValue);
      inputRef.current?.blur();
    } else if (e.key === "Escape") {
      setInputValue(formattedValue); // Revert
      isTyping.current = false;
      inputRef.current?.blur();
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const direction = e.key === "ArrowUp" ? 1 : -1;
      const increment = e.shiftKey ? config.step * 10 : config.step; // Match original stepping
      let currentNum = parseFloat(inputValue);
      if (isNaN(currentNum)) currentNum = params[paramKey] as number; // Fallback to state value

      const newValue = roundToPrecision(currentNum + direction * increment, 4); // Round intermediate step
      const clampedNewValue = Math.max(
        config.min,
        Math.min(config.max, newValue)
      );

      // Use flushSync to commit immediately for better responsiveness with arrow keys
      flushSync(() => {
        onParamChange(paramKey, clampedNewValue);
      });
      // Update input directly after state update (use formatted value from potentially updated state)
      const nextFormattedValue = formatValue
        ? formatValue(clampedNewValue)
        : clampedNewValue.toFixed(3);
      setInputValue(nextFormattedValue);
      isTyping.current = true; // Keep typing mode active

      // Select text after update
      requestAnimationFrame(() => inputRef.current?.select());
    }
  };

  return (
    <>
      <div>
        <Label htmlFor={paramKey} className="pr-4 text-nowrap">
          {label}
        </Label>
      </div>
      <div>
        <Slider
          id={paramKey + "-slider"}
          min={config.min}
          max={config.max}
          step={config.step}
          value={[value]}
          onValueChange={handleSliderChange}
          className="h-8" // Match original slider track height/centering
        />
      </div>
      <div className="max-sm:hidden">
        <Input
          ref={inputRef}
          id={paramKey}
          type="text" // Use text to allow intermediate non-numeric input
          inputMode="decimal" // Hint for mobile keyboards
          value={inputValue}
          onKeyDown={handleInputKeyDown}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onFocus={e => e.target.select()}
          className="h-10 w-full rounded bg-white/15 pl-3 text-sm tabular-nums outline-white/20 focus:outline-2 focus:-outline-offset-1 focus:outline-blue-500" // Style like original numeric input
          autoCapitalize="none"
          autoComplete="off"
          autoCorrect="off"
          spellCheck="false"
          data-1p-ignore="true" // Password manager ignore hints
          data-lpignore="true"
          data-bwignore="true"
          data-form-type="other"
        />
      </div>
    </>
  );
}

// --- Main Hero Component ---
export default function HeroPage({
  params: { imageId },
}: {
  params: { imageId: string };
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [params, setParams] = useState<typeof defaultParams>(() => {
    // Initialize state from URL or defaults
    const initial: typeof defaultParams = { ...defaultParams };
    let updatedFromUrl = false;
    for (const key in paramConfig) {
      if (searchParams.has(key)) {
        const urlValue = searchParams.get(key);
        const numValue = parseFloat(urlValue || "");
        if (!isNaN(numValue)) {
          initial[key as keyof typeof defaultParams] = numValue;
          updatedFromUrl = true;
        }
      }
    }
    if (searchParams.has("background")) {
      initial.background = searchParams.get("background") || "metal";
      updatedFromUrl = true;
    }
    return initial;
  });
  const [imageData, setImageData] = useState<ImageData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const latestParamsRef = useRef(params); // Ref to prevent stale closures in debounced function

  // Ref to track if initial URL load has set state, preventing immediate overwrite
  const initialUrlLoadDone = useRef(false);

  // Fetch and process initial image
  useEffect(() => {
    setIsLoading(true);
    const fetchImage = async () => {
      try {
        // Vercel Blob URL structure - adjust if different
        // const imageUrl = `https://p1ljtcp1ptfohfxm.public.blob.vercel-storage.com/${imageId}.png`;
        const imageUrl = "/apple.svg";

        const response = await fetch(imageUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch image: ${response.statusText}`);
        }
        const blob = await response.blob();
        const file = new File([blob], `${imageId}.png`, { type: blob.type });

        const processed = await processImageForShader(file);
        if (processed) {
          setImageData(processed.imageData);
        } else {
          toast.error("Failed to process initial image.");
        }
      } catch (error) {
        console.error("Error loading initial image:", error);
        toast.error(
          `Error loading image: ${error instanceof Error ? error.message : "Unknown error"}`
        );
        // Maybe redirect or show an error state
      } finally {
        setIsLoading(false);
        initialUrlLoadDone.current = true; // Mark initial load complete AFTER fetching
      }
    };
    fetchImage();
  }, [imageId]); // Only refetch if imageId changes

  // Update URL params when state changes (debounced)
  const debouncedUpdateUrl = useCallback(
    debounce(() => {
      const currentParams = latestParamsRef.current;
      const newSearchParams = new URLSearchParams();
      Object.entries(currentParams).forEach(([key, value]) => {
        if (key !== "background") {
          newSearchParams.set(
            key,
            roundToPrecision(value as number, 4).toString()
          );
        } else {
          newSearchParams.set(key, value as string);
        }
      });
      // Use replace to avoid adding to history stack for every param change
      router.replace(`${pathname}?${newSearchParams.toString()}`, {
        scroll: false,
      });
    }, 250),
    [pathname, router]
  ); // Dependencies for router/pathname

  // Effect to sync state to URL
  useEffect(() => {
    latestParamsRef.current = params;
    // Only update URL if the initial load is done and state wasn't just set FROM the URL
    if (initialUrlLoadDone.current) {
      debouncedUpdateUrl();
    }
  }, [params, debouncedUpdateUrl]);

  // Effect to sync URL to state (on initial load/navigation)
  useEffect(() => {
    const newParamsFromUrl: Partial<typeof defaultParams> = {};
    let changed = false;
    for (const key in paramConfig) {
      if (searchParams.has(key)) {
        const urlValue = searchParams.get(key);
        const numValue = parseFloat(urlValue || "");
        const currentNumValue = roundToPrecision(
          params[key as keyof typeof paramConfig] as number,
          4
        );
        if (
          !isNaN(numValue) &&
          roundToPrecision(numValue, 4) !== currentNumValue
        ) {
          newParamsFromUrl[key as keyof typeof paramConfig] = numValue;
          changed = true;
        }
      }
    }
    const urlBackground = searchParams.get("background");
    if (urlBackground && urlBackground !== params.background) {
      newParamsFromUrl.background = urlBackground;
      changed = true;
    }

    if (changed) {
      console.log("Updating state from URL params");
      setParams(prev => ({ ...prev, ...newParamsFromUrl }));
    }
    // Mark initial load as done here if not handled by image fetch completion
    // initialUrlLoadDone.current = true;
  }, [searchParams]); // Rerun when searchParams change

  const handleParamChange = useCallback(
    (key: keyof typeof defaultParams, value: number | string) => {
      setParams(prev => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleFileSelect = (files: FileList | null) => {
    if (files && files.length > 0) {
      const file = files[0];
      if (file.size > 4.5 * 1024 * 1024) {
        // ~4.5MB limit
        toast.error("File size must be less than 4.5MB");
        return;
      }
      if (!file.type.startsWith("image/") && file.type !== "image/svg+xml") {
        toast.error("Please upload only images or SVG files.");
        return;
      }

      setIsLoading(true); // Show loading for processing
      setIsUploading(true); // Show uploading indicator

      processImageForShader(file)
        .then(async processed => {
          if (processed) {
            setImageData(processed.imageData);
            // Upload the processed PNG blob
            try {
              const uploadResponse = await fetch("/api/user-logo", {
                // Your API endpoint
                method: "POST",
                headers: { "Content-Type": "image/png" },
                body: processed.pngBlob,
              });
              if (!uploadResponse.ok) {
                throw new Error(
                  `Upload failed: ${uploadResponse.status} ${await uploadResponse.text()}`
                );
              }
              const { imageId: newImageId } = await uploadResponse.json();
              // Update URL to reflect the new image ID (using replace)
              router.replace(
                `/share/${newImageId}?${searchParams.toString()}`,
                { scroll: false }
              );
              toast.success("Image uploaded successfully!");
            } catch (uploadError) {
              console.error("Error uploading image:", uploadError);
              toast.error(
                `Error uploading image: ${uploadError instanceof Error ? uploadError.message : "Unknown error"}`
              );
            }
          }
        })
        .finally(() => {
          setIsLoading(false);
          setIsUploading(false);
        });
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(event.target.files);
    event.target.value = ""; // Reset input
  };

  // Drag and Drop Handlers
  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    // Check if leaving to outside the window or relatedTarget is null/outside the component
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  };
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); // Necessary to allow drop
    e.stopPropagation();
    setIsDragging(true); // Keep true while over
  };
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const backgroundStyle: React.CSSProperties = useMemo(() => {
    if (params.background === "metal") {
      return { background: "linear-gradient(to bottom, #eee, #b8b8b8)" };
    }
    return { background: params.background };
  }, [params.background]);

  return (
    // Main container with drag/drop handlers
    <div
      className="flex bg-foreground flex-col items-center justify-center min-h-screen gap-8 px-4 py-8 md:px-8 lg:flex-row lg:items-start lg:gap-8 relative" // Added relative for potential overlay
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Drag Overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-10 flex items-center justify-center text-xl font-semibold text-white bg-black/50 pointer-events-none rounded-lg">
          Drop image here
        </div>
      )}

      {/* Visual Output Area */}
      <div
        className="flex aspect-square w-full max-w-[500px] items-center justify-center rounded-lg shadow-lg"
        style={backgroundStyle}
      >
        <div className="relative aspect-square size-full">
          {" "}
          {/* Fixed size container */}
          {isLoading && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/30">
              <div className="w-12 h-12 border-4 border-t-transparent border-white rounded-full animate-spin"></div>
              {isUploading && (
                <span className="ml-4 text-white">Processing...</span>
              )}
            </div>
          )}
          <Canvas camera={{ position: [0, 0, 1.5], fov: 80 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            {imageData && (
              <ShaderEffect imageData={imageData} params={params} />
            )}
          </Canvas>
        </div>
      </div>

      {/* Controls Area */}
      <div className="grid w-full max-w-[500px] auto-rows-min grid-cols-[auto_1fr] sm:grid-cols-[auto_1fr_100px] items-center gap-x-6 gap-y-3 rounded-lg p-4 bg-foreground text-white shadow-lg outline outline-white/20">
        {/* Background Control */}
        <div>
          <Label className="pr-4 text-nowrap">Background</Label>
        </div>
        <div className="flex h-10 items-center gap-2 sm:col-span-2">
          {/* Background Buttons */}
          <Button
            variant="outline"
            size="icon"
            className="w-7 h-7 rounded-full"
            style={{ background: "linear-gradient(to bottom, #eee, #b8b8b8)" }}
            onClick={() => handleParamChange("background", "metal")}
            aria-label="Metal Background"
          />
          <Button
            variant="outline"
            size="icon"
            className="w-7 h-7 rounded-full bg-white"
            onClick={() => handleParamChange("background", "white")}
            aria-label="White Background"
          />
          <Button
            variant="outline"
            size="icon"
            className="w-7 h-7 rounded-full bg-black border border-white/30"
            onClick={() => handleParamChange("background", "black")}
            aria-label="Black Background"
          />
          {/* Color Picker */}
          <Label
            className={cn(
              "relative w-7 h-7 cursor-pointer rounded-full text-[0px] focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500 focus-within:ring-offset-gray-800",
              // Conic gradient background for picker representation
              `bg-[radial-gradient(circle,white,transparent_65%),conic-gradient(in_oklch,oklch(63%_.25_30),oklch(79%_.17_70),oklch(97%_.21_110),oklch(87%_.24_150),oklch(90%_.16_190),oklch(76%_.15_230),oklch(47%_.31_270),oklch(60%_.30_310),oklch(66%_.28_350),oklch(63%_.25_30))]`
            )}
          >
            <Input
              type="color"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              value={
                typeof params.background === "string" &&
                params.background.startsWith("#")
                  ? params.background
                  : "#ffffff"
              } // Handle non-color string values
              onChange={e => handleParamChange("background", e.target.value)}
              aria-label="Custom Background Color"
            />
            Custom
          </Label>
        </div>

        {/* Parameter Sliders/Inputs */}
        {Object.entries(paramConfig).map(([key, config]) => (
          <ParamControl
            key={key}
            label={
              key.charAt(0).toUpperCase() +
              key.slice(1).replace(/([A-Z])/g, " $1")
            } // Format label
            paramKey={key as keyof typeof defaultParams}
            params={params}
            config={config}
            onParamChange={handleParamChange}
            // Custom format for patternScale
            formatValue={
              key === "patternScale"
                ? v => (v === 0 || v === 10 ? v.toString() : v.toFixed(1))
                : undefined
            }
          />
        ))}

        {/* Upload Button and Tip */}
        <div className="col-span-full mt-4">
          <Button asChild className="w-full mb-4 h-10 select-none">
            <Label htmlFor="file-input" className="cursor-pointer">
              Upload Image
            </Label>
          </Button>
          <Input
            type="file"
            accept="image/*,.svg"
            onChange={handleFileChange}
            id="file-input"
            className="hidden"
          />
          <p className="w-full text-xs text-white/80">
            Tips: Transparent or white background works best. Shapes are better
            than text. Use SVG or high-res images (500px-1000px recommended).
            Max 4.5MB.
          </p>
        </div>
      </div>
    </div>
  );
}
