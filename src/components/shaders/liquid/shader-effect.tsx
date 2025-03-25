import { useEffect, useMemo, useRef } from "react";
import { vertexShader } from "./vertex-shader";
import { fragmentShader } from "./fragment-shader";
import { shaderMaterial } from "@react-three/drei";
import { extend, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { defaultParams } from "./param-controls";

// --- Shader Component ---
interface ShaderEffectProps {
  imageData: ImageData | null;
  params: typeof defaultParams;
}

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

export function ShaderEffect({ imageData, params }: ShaderEffectProps) {
  const shaderRef = useRef<THREE.ShaderMaterial>(null);
  const texture = useMemo(() => {
    return imageData ? new THREE.CanvasTexture(imageData) : null; // Use CanvasTexture for ImageData
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
      {/* @ts-expect-error type added to globals  */}
      <liquidShaderMaterial
        ref={shaderRef}
        key={LiquidShaderMaterial.key} // Needed for HMR
        transparent={true}
        blending={THREE.NormalBlending}
      />
    </mesh>
  );
}
