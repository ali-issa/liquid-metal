// Extend THREE namespace for JSX shaderMaterial
declare global {
  namespace JSX {
    interface IntrinsicElements {
      liquidShaderMaterial: JSX.IntrinsicElements["meshStandardMaterial"] & {
        key?: string;
        ref?: React.RefObject<THREE.ShaderMaterial>;
      };
    }
  }
}
