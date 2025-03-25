"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Canvas } from "@react-three/fiber";

// Shadcn UI Components
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { toast } from "sonner"; // Assuming sonner for toast notifications

// Tailwind utility
import { cn } from "@/lib/utils";
import { defaultParams, paramConfig, ParamControl } from "./param-controls";
import { processImageForShader, roundToPrecision } from "./helpers";
import { debounce } from "@/utils/debounce";
import { ShaderEffect } from "./shader-effect";

// --- Main Hero Component ---
export function LiquidMetal({ imageId }: { imageId: string }) {
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
          // @ts-expect-error number
          initial[key as keyof typeof defaultParams] = numValue;
          updatedFromUrl = true;
        }
      }
    }
    if (searchParams.has("background")) {
      initial.background = searchParams.get("background") || "metal";
      // eslint-disable-next-line
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
          `Error loading image: ${
            error instanceof Error ? error.message : "Unknown error"
          }`
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
                  `Upload failed: ${
                    uploadResponse.status
                  } ${await uploadResponse.text()}`
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
                `Error uploading image: ${
                  uploadError instanceof Error
                    ? uploadError.message
                    : "Unknown error"
                }`
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
