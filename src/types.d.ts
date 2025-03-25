// Extend THREE namespace for JSX shaderMaterial
declare global {
  namespace JSX {
    interface IntrinsicElements {
      liquidShaderMaterial: any; // Adjust type if possible
    }
  }
}
