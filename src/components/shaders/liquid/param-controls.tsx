import { useEffect, useRef, useState } from "react";
import { roundToPrecision } from "./helpers";
import { flushSync } from "react-dom";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

// --- UI Component for a single Parameter Slider + Input ---
interface ParamControlProps {
  label: string;
  paramKey: keyof typeof defaultParams;
  params: typeof defaultParams;
  config: { min: number; max: number; step: number; default: number };
  onParamChange: (key: keyof typeof defaultParams, value: number) => void;
  formatValue?: (value: number) => string;
}

// --- Default Params ---
export const paramConfig = {
  refraction: { min: 0, max: 0.06, step: 0.001, default: 0.015 },
  edge: { min: 0, max: 1, step: 0.01, default: 0.4 },
  patternBlur: { min: 0, max: 0.05, step: 0.001, default: 0.005 },
  liquid: { min: 0, max: 1, step: 0.01, default: 0.07 },
  speed: { min: 0, max: 1, step: 0.01, default: 0.3 },
  patternScale: { min: 1, max: 10, step: 0.1, default: 2 },
};

// @ts-expect-error type
export const defaultParams = Object.fromEntries(
  Object.entries(paramConfig).map(([key, config]) => [key, config.default])
) as Record<keyof typeof paramConfig, number> & { background: string };
defaultParams.background = "metal"; // Add background default

export function ParamControl({
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
