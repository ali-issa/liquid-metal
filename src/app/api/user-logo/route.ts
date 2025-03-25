import { NextResponse } from "next/server";
import { put } from "@vercel/blob";
import { nanoid } from "nanoid";

export async function POST(request: Request): Promise<NextResponse> {
  const blob = await request.blob();

  if (blob.type !== "image/png") {
    return NextResponse.json(
      { message: "Invalid file type. Only PNG allowed." },
      { status: 400 }
    );
  }
  if (blob.size > 4.5 * 1024 * 1024) {
    // ~4.5MB limit server-side too
    return NextResponse.json(
      { message: "File size exceeds 4.5MB limit." },
      { status: 413 }
    );
  }

  try {
    const imageId = nanoid(16); // Generate a unique ID
    const filename = `${imageId}.png`;

    // Upload to Vercel Blob (or your storage provider)
    const blobResult = await put(filename, blob, {
      access: "public",
      contentType: "image/png",
      // Add any cache control headers if needed
      //   cacheControlMaxAge: 31536000, // e.g., 1 year
    });

    // Return the URL and the generated ID
    return NextResponse.json({ imageId: imageId, url: blobResult.url });
  } catch (error) {
    console.error("Error uploading to blob storage:", error);
    return NextResponse.json(
      { message: "Failed to upload image." },
      { status: 500 }
    );
  }
}
