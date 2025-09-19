// File: src/components/dashboard/SidebarSpec.tsx

const SECTIONS = [
  {
    label: "AMBIENT",
    background: "#39c06b",
    items: [
      { target: "FRONT#1", spec: "Spec: +/- 17 ms" },
      { target: "FRONT#2", spec: "Spec: +/- 20 ms" },
      { target: "REAR#3", spec: "Spec: +/- 19 ms" },
    ],
  },
  {
    label: "HOT",
    background: "#e23b2e",
    items: [
      { target: "FRONT#1", spec: "Spec: +/- 17 ms" },
      { target: "FRONT#2", spec: "Spec: +/- 20 ms" },
      { target: "REAR#3", spec: "Spec: +/- 19 ms" },
    ],
  },
  {
    label: "COLD",
    background: "#3fa0f5",
    items: [
      { target: "FRONT#1", spec: "Spec: +/- 21 ms" },
      { target: "FRONT#2", spec: "Spec: +/- 22 ms" },
      { target: "REAR#3", spec: "Spec: +/- 20 ms" },
    ],
  },
] satisfies Array<{
  label: string;
  background: string;
  items: Array<{ target: string; spec: string }>;
}>;

export default function SidebarSpec() {
  return (
    <div className="space-y-4 text-white">
      <div className="mx-auto w-full max-w-[160px] text-center">
        <div className="text-[32px] font-black tracking-[0.4px]">Autoliv</div>
        <div className="mt-1 h-[7px] w-full bg-white" />
      </div>

      <div className="text-center text-[25px] font-black uppercase tracking-[0.8px]">
        SPEC
      </div>
      <div className="mx-1 h-1 rounded bg-white/35" />

      {SECTIONS.map((section, idx) => (
        <div key={section.label} className="space-y-2">
          <div className="text-center text-lg font-black tracking-[0.6px]">
            {section.label}
          </div>
          <div className="grid grid-cols-3 gap-2">
            {section.items.map((item) => (
              <div
                key={item.target}
                className="flex h-[68px] flex-col items-center justify-center rounded-xl px-1.5 py-1.5 text-center text-[11px] font-extrabold leading-tight text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.6),0_8px_18px_rgba(0,0,0,0.25)]"
                style={{ background: section.background }}
              >
                <div>Target</div>
                <div>{item.target}</div>
                <small className="mt-1 block text-[11px] font-bold text-[#f9f62c]">
                  {item.spec}
                </small>
              </div>
            ))}
          </div>
          {idx < SECTIONS.length - 1 && <div className="mx-1 mt-4 h-1 rounded bg-white/35" />}
        </div>
      ))}
    </div>
  );
}
