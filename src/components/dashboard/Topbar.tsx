import React from 'react';

export default function Topbar() {
  return (
    <div
      className="bg-transparent border-0 shadow-none flex items-center p-3.5 min-h-[64px] justify-center"
    >
      <div className="flex flex-col items-center gap-1 text-center">
        <h1 className="m-0 text-[37px] font-extrabold tracking-wider uppercase text-[#164799] leading-tight">
          Airbag Detection Dashboard
        </h1>
        <span className="text-[#164799] font-bold text-[20px]">
          Real-time monitoring and alerts
        </span>
      </div>
    </div>
  );
}
