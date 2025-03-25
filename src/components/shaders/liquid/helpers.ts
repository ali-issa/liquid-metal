import { toast } from "sonner";

// This function needs careful porting and testing. It's complex.
export async function processImageForShader(
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

// --- Helper: Round Number ---
// From original: v = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
const roundFactors = [1, 10, 100, 1000, 10000, 100000, 1000000];
export function roundToPrecision(value: number, precision: number = 2): number {
  const factor = roundFactors[precision] ?? Math.pow(10, precision);
  const temp = value * factor;
  const roundedTemp = Math.trunc(temp + (temp > 0 ? 0.5 : -0.5));
  const result = roundedTemp / factor;
  return result === 0 ? 0 : result; // Handle -0 case
}
