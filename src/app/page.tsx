import { LiquidMetal } from "@/components/shaders/liquid/liquid-metal";
import { Suspense } from "react";

export default async function Page({
  params,
}: {
  params: Promise<{ imageId: string }>;
}) {
  const { imageId } = await params;

  return (
    <Suspense>
      <LiquidMetal imageId={imageId} />
    </Suspense>
  );
}
