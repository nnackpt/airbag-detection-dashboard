"use client";
import React, { useEffect, useRef } from "react";
import type { Alert } from "@/types/airbag";

type Props = {
  alert: Alert;
  onContinue: (id: string) => Promise<void> | void;
  onClose?: () => void;
};

export default function NgAlertModal({ alert, onContinue, onClose }: Props) {
  const continueRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    continueRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose?.();
      if (e.key === "Enter") onContinue(alert.id);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [alert.id, onClose, onContinue]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="ngv2-title"
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-2 sm:p-4 backdrop-blur-sm"
      style={{ animation: "fadeIn 0.3s ease-out" }}
    >
      <div
        role="document"
        className="relative w-full max-w-[min(92vw,48rem)] bg-white shadow-2xl flex max-h-[90vh] flex-col rounded-2xl overflow-hidden border border-red-100"
        style={{ 
          // minWidth: '700px',
          animation: "slideUp 0.3s ease-out",
          boxShadow: "0 25px 50px -12px rgba(220, 38, 38, 0.25)"
        }}
      >
        {/* Close Button - Modern floating style */}
        <button
          aria-label="Close"
          onClick={onClose}
          className="absolute right-4 top-4 z-10 inline-flex h-10 w-10 items-center justify-center text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-red-300 focus:ring-offset-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Header Section with Gradient Background */}
        <div className="bg-gradient-to-br from-red-500 via-red-600 to-red-700 px-6 py-4 relative overflow-hidden">
          {/* Background Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute inset-0" style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='7' cy='7' r='1'/%3E%3Ccircle cx='27' cy='7' r='1'/%3E%3Ccircle cx='47' cy='7' r='1'/%3E%3Ccircle cx='7' cy='27' r='1'/%3E%3Ccircle cx='27' cy='27' r='1'/%3E%3Ccircle cx='47' cy='27' r='1'/%3E%3Ccircle cx='7' cy='47' r='1'/%3E%3Ccircle cx='27' cy='47' r='1'/%3E%3Ccircle cx='47' cy='47' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
            }} />
          </div>

          {/* Header Content */}
          <div className="relative flex items-center justify-center gap-4">
            {/* Warning Icon with Pulse Animation */}
            <div className="relative">
              <div className="w-14 h-14 bg-white rounded-full flex items-center justify-center shadow-lg">
                <svg viewBox="0 0 64 64" className="w-8 h-8" aria-hidden="true">
                  <path
                    d="M31 7c1.1-1.9 3.9-1.9 5 0l25 43c1.1 2-0.3 4.5-2.5 4.5H8.5C6.3 54.5 4.9 52 6 50z"
                    className="fill-red-500"
                  />
                  <path
                    d="M31.9 18h2.2l1.6 21h-5.4l1.6-21zm-0.1 26c0-1.7 1.3-3 3-3s3 1.3 3 3-1.3 3-3 3-3-1.3-3-3z"
                    className="fill-white"
                  />
                </svg>
              </div>
              {/* Pulse rings */}
              <div className="absolute inset-0 rounded-full bg-white animate-ping opacity-10"></div>
              <div className="absolute inset-0 rounded-full bg-white animate-pulse opacity-20"></div>
            </div>

            {/* NG Title with Modern Typography */}
            <div className="text-center">
              <div
                id="ngv2-title"
                className="select-none font-black text-white text-[clamp(28px,6vw,40px)] tracking-widest drop-shadow-lg"
                style={{ textShadow: "0 4px 8px rgba(0,0,0,0.3)" }}
              >
                NG
              </div>
              <div className="text-red-100 text-xs md:text-sm font-medium mt-1 tracking-wide uppercase">
                Quality Alert
              </div>
            </div>
          </div>
        </div>

        {/* Status Message Bar */}
        <div className="bg-gradient-to-r from-red-600 to-red-700 px-6 py-2 text-center border-b border-red-500/20">
          <div className="flex items-center justify-center gap-2 text-white">
            <svg className="w-5 h-5 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <span className="text-lg font-medium">
              ระบบจะหยุดการประมวลผลชั่วคราวจนกว่าผู้ใช้จะกด
              <span className="mx-2 font-bold underline underline-offset-2 decoration-2 decoration-red-200">
                Continue
              </span>
            </span>
          </div>
        </div>

        {/* Main Content */}
        <section className="flex-1 overflow-y-auto bg-gradient-to-b from-gray-50 to-gray-100">
          <div className="p-4 space-y-4">
            {/* Video Info Card */}
            <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-gray-700 to-gray-800 px-4 py-2.5">
                <h3 className="text-white font-semibold text-lg flex items-center gap-2">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M2 6a2 2 0 012-2h6l2 2h6a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
                  </svg>
                  Video Information
                </h3>
              </div>
              <div className="p-4 space-y-2.5">
                <div className="flex items-center gap-3">
                  <span className="font-medium text-gray-600 min-w-20 text-sm">Name:</span>
                  <span className="text-gray-900 font-semibold bg-gray-100 px-3 py-1 rounded-lg">
                    {alert.video_name}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-medium text-gray-600 min-w-20 text-sm">SN:</span>
                  <span className="text-gray-900 font-mono bg-blue-50 text-blue-800 px-3 py-1 rounded-lg">
                    {alert.module_sn}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-medium text-gray-600 min-w-20 text-sm">Temp:</span>
                  <span className="text-gray-900 bg-orange-50 text-orange-800 px-3 py-1 rounded-lg">
                    {alert.temperature_type}
                  </span>
                </div>
              </div>
            </div>

            {/* Test Results Table */}
            <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
              <div className="bg-gradient-to-r from-red-600 to-red-700 px-4 py-2.5">
                <h3 className="text-white font-semibold text-lg flex items-center gap-2">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zm0 4a1 1 0 011-1h6a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h6a1 1 0 110 2H4a1 1 0 01-1-1zm8-4a1 1 0 011-1h4a1 1 0 110 2h-4a1 1 0 01-1-1zm0 4a1 1 0 011-1h4a1 1 0 110 2h-4a1 1 0 01-1-1z" clipRule="evenodd" />
                  </svg>
                  Test Results
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-gradient-to-r from-gray-100 to-gray-200 border-b border-gray-300">
                    <tr>
                      <th className="px-4 py-2 font-bold text-gray-800 text-xs md:text-sm uppercase tracking-wider">Parameter</th>
                      <th className="px-4 py-2 font-bold text-gray-800 text-xs md:text-sm uppercase tracking-wider">Measured</th>
                      <th className="px-4 py-2 font-bold text-gray-800 text-xs md:text-sm uppercase tracking-wider">Spec</th>
                      <th className="px-4 py-2 font-bold text-gray-800 text-xs md:text-sm uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {alert.details.map((d, i) => {
                      const st = (d.status || "").toUpperCase();
                      const isNG = st === "NG";
                      const isOK = st === "OK";

                      return (
                        <tr key={i} className={`hover:bg-gray-50 transition-colors duration-200 ${
                          isNG ? 'bg-red-50/50' : isOK ? 'bg-green-50/50' : ''
                        }`}>
                          <td className="px-4 py-2 text-gray-900 font-medium">
                            {d.parameter}
                          </td>
                          <td className="px-4 py-2 text-gray-800 font-mono text-xs md:text-sm">
                            {d.value ?? "-"}
                          </td>
                          <td className="px-4 py-2 text-gray-600 font-mono text-xs md:text-sm">
                            {d.spec_limit ?? "-"}
                          </td>
                          <td className="px-4 py-2">
                            {st ? (
                              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-bold uppercase tracking-wide ${
                                isNG
                                  ? "bg-red-100 text-red-800 border border-red-200"
                                  : isOK
                                  ? "bg-green-100 text-green-800 border border-green-200"
                                  : "bg-gray-100 text-gray-600 border border-gray-200"
                              }`}>
                                {st}
                              </span>
                            ) : (
                              <span className="text-gray-400">-</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Additional Message */}
            {alert.message && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <h4 className="text-amber-800 font-semibold mb-1">Additional Information</h4>
                    <p className="text-amber-700">{alert.message}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Footer Actions */}
        <footer className="flex items-center justify-end gap-3 bg-white px-4 py-3 border-t border-gray-200">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm font-semibold text-gray-700 bg-white border-2 border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-300 focus:ring-offset-2 transition-all duration-200"
          >
            Close
          </button>
          <button
            type="button"
            ref={continueRef}
            onClick={() => onContinue(alert.id)}
            className="px-6 py-2 text-sm font-bold text-white bg-gradient-to-r from-green-600 to-green-700 rounded-lg hover:from-green-700 hover:to-green-800 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            Continue
          </button>
        </footer>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from { 
            opacity: 0; 
            transform: translateY(20px) scale(0.95); 
          }
          to { 
            opacity: 1; 
            transform: translateY(0) scale(1); 
          }
        }
      `}</style>
    </div>
  );
}