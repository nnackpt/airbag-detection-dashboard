import { useCallback, useEffect, useMemo, useRef, useState } from "react";
// import Image from "next/image";
import { ProcessingStatus as ProcessingStatusType } from "@/types/airbag";
import { FolderImageKey, getFolderImages } from "@/services/StatsApi";
import { API_BASE } from "@/services/airbagApi";

type ImgState = Partial<Record<FolderImageKey, string>>;

const LABELS: Record<FolderImageKey, string> = {
  explosion: "Explosion",
  fr1: "Front #1",
  fr2: "Front #2",
  re3: "Rear #3",
  full_deployment: "Full",
};
const ORDER: FolderImageKey[] = [
  "explosion",
  "fr1",
  "fr2",
  "re3",
  "full_deployment",
];

// Only report *why* we fallback
function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

export default function ProcessingStatusBox({
  item,
}: {
  item: ProcessingStatusType | null;
}) {
  const rawProgress = Number(item?.progress ?? 0);
  const progress = Math.max(
    0,
    Math.min(100, Number.isFinite(rawProgress) ? rawProgress : 0)
  );
  const compact = progress >= 100;
  const start = item?.start_time
    ? new Date(item.start_time).toLocaleString("en-US")
    : "N/A";

  const candidates = useMemo(() => {
    const rawName = (item?.video_name || "").split(/[\\/]/).pop() || "";
    const noExt = rawName.replace(/\.[^.]+$/, "");
    const normalized = rawName.replace(/[.\s]+/g, "_");

    // เพิ่มตัวเลือกที่เก็บช่องว่างไว้
    const withSpaces = rawName.replace(/[.]+/g, "_"); // แทนที่แค่จุด ไม่แทนที่ช่องว่าง

    return Array.from(
      new Set([
        noExt,
        rawName,
        normalized,
        withSpaces, // เพิ่มตัวเลือกนี้
        rawName.trim(), // ลบช่องว่างหน้าหลัง
      ])
    ).filter(Boolean);
  }, [item?.video_name]);

  const [imgs, setImgs] = useState<ImgState>({});
  const [folderUsed, setFolderUsed] = useState<string>("");
  const [errorHint, setErrorHint] = useState<string>("");
  const [loadingImgs, setLoadingImgs] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Normalize absolute URL to share origin with API_BASE
  const toAbsUrl = (u?: string) => {
    if (!u) return undefined;
    try {
      const base = new URL(API_BASE);
      // ไม่ต้อง encode ซ้ำเพราะ API ส่งมาแล้ว encoded แล้ว
      const abs = new URL(u, base.origin);
      return abs.toString();
    } catch (error) {
      console.error("Failed to create absolute URL:", u, error);
      return u.startsWith("http") ? u : `${API_BASE}${u}`;
    }
  };

  // Stable delay to keep deps simple
  const pollDelay = progress < 100 ? 2000 : 5000;

  // Stable fetcher to satisfy exhaustive-deps
  const fetchImages = useCallback(async () => {
    setLoadingImgs(true);
    setErrorHint("");
    for (const name of candidates) {
      try {
        const res = await getFolderImages(name);
        const images = res.images || {};

        // เพิ่มการตรวจสอบ URL ของรูปภาพ
        const validImages: ImgState = {};
        for (const [key, url] of Object.entries(images)) {
          if (url && typeof url === "string") {
            validImages[key as FolderImageKey] = url;
          }
        }

        if (Object.keys(validImages).length > 0) {
          setImgs(validImages);
          setFolderUsed(name);
          setLoadingImgs(false);

          setStickyImgs(validImages);
          setStickyFolder(name);
          if (typeof window !== "undefined") {
            try {
              localStorage.setItem("lastKeyFrames", JSON.stringify(validImages));
              localStorage.setItem("lastKeyFramesFolder", name);
            } catch {}
          }

          console.log("Found images for folder:", name, validImages);
          return;
        } else {
          setErrorHint(`Found folder "${name}" but images not ready yet`);
        }
      } catch (err) {
        const msg = getErrorMessage(err);
        console.error("getFolderImages error:", name, msg);
        setErrorHint(`Cannot fetch images for "${name}" (${msg})`);
      }
    }
    setLoadingImgs(false);
  }, [candidates]);

  useEffect(() => {
    // setImgs({});
    setFolderUsed("");
    setErrorHint("");

    if (candidates.length === 0) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      await fetchImages();
    };
    run();

    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(fetchImages, pollDelay);

    return () => {
      cancelled = true;
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [candidates, pollDelay, fetchImages]);

  useEffect(() => {
    const haveAll = ORDER.every((k) => !!imgs?.[k]);
    if (haveAll && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, [imgs]);

  const [stickyImgs, setStickyImgs] = useState<ImgState>(() => {
    if (typeof window !== "undefined") {
      try {
        const raw = localStorage.getItem("lastKeyFrames");
        return raw ? (JSON.parse(raw) as ImgState) : {};
      } catch {
        return {};
      }
    }
    return {};
  });

  const [stickyFolder, setStickyFolder] = useState<string>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("lastKeyFramesFolder") || "";
    }
    return "";
  });

  // ใช้ตัวแปรแสดงผล: ถ้า imgs ว่าง ให้ใช้ stickyImgs/ stickyFolder แทน
  const displayImgs = Object.keys(imgs || {}).length ? imgs : stickyImgs;
  const displayFolderUsed = folderUsed || stickyFolder;

  return (
    <section
      className="rounded-none bg-[#164799] text-white border border-white/30 shadow-[0_12px_28px_rgba(10,26,48,0.16)] p-4 min-h-[120px]"
      id="processingBox"
    >
      <h2
        className={`text-[22px] font-black mb-2 ${
          compact ? "text-[#28c26a]" : "text-white"
        }`}
      >
        {compact ? "COMPLETED" : "PROCESSING STATUS"}
      </h2>

      <div className="font-extrabold text-base opacity-95 mb-4" id="procName">
        {item?.video_name ?? "Queue is empty"}
      </div>

      <div className="h-10 overflow-hidden border-2 border-black/15 bg-[#d4d9df]">
        <div
          className="h-full bg-[#28c26a] transition-all duration-500"
          id="procFill"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div
        className={`inline-flex items-center justify-center font-black text-white bg-[#28c26a] border-[4px] border-[#1f9d55] rounded-[20px] mt-3 ${
          compact ? "text-lg px-4 py-1" : "text-[40px] px-4 py-1"
        }`}
        id="procPct"
      >
        {progress}%
      </div>

      <div className="mt-2 text-sm font-black opacity-90" id="procStart">
        Start: {start}
      </div>

      <div className="mt-4">
        {/* <div className="text-xs font-extrabold opacity-90 mb-2">
          KEY FRAMES
          {loadingImgs ? " · Loading..." : ""}
          {displayFolderUsed ? ` · Folder: ${displayFolderUsed}` : ""}
          {errorHint ? ` · ${errorHint}` : ""}
        </div> */}

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          {ORDER.map((k) => {
            const src = toAbsUrl(displayImgs?.[k]);
            const label = LABELS[k];
            return (
              <div
                key={k}
                className="bg-white/5 border border-white/20 rounded-md p-2 flex flex-col"
              >
                <div className="text-[10px] font-black uppercase mb-1 tracking-wide opacity-90">
                  {label}
                </div>

                {src ? (
                  <a
                    href={src}
                    target="_blank"
                    rel="noreferrer"
                    className="relative block"
                  >
                    <div className="w-full aspect-video relative">
                      <img
                        src={src}
                        alt={label}
                        className="w-full h-full object-cover rounded-[8px] ring-2 ring-white/30 hover:ring-white transition"
                        onError={(e) => {
                          console.error("IMG load failed:", src);
                          // Show placeholder instead of hiding
                          e.currentTarget.src =
                            "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzMzMyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmaWxsPSIjZmZmIiBkb21pbmFudC1iYXNlbGluZT0ibWlkZGxlIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXNpemU9IjEycHgiPkVycm9yPC90ZXh0Pjwvc3ZnPg==";
                        }}
                        onLoad={() => {
                          console.log("IMG loaded successfully:", src);
                        }}
                      />
                    </div>
                  </a>
                ) : (
                  <div className="w-full aspect-video rounded-[8px] bg-black/30 backdrop-blur-sm flex items-center justify-center text-[10px] opacity-70">
                    No Image Available
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
