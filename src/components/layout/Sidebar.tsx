"use client"

import Link from "next/link";
import React, { useRef } from "react";
import { usePathname } from "next/navigation";

import {
  ChartBarIcon,
  ArrowUpIcon,
  HomeIcon,
} from "@heroicons/react/24/outline";

import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { Check, ChevronDown, PanelLeftClose, PanelLeftOpen } from "lucide-react";

// Types
interface MenuItem {
  label: string;
  href: string;
  icon?: React.ComponentType<{ className?: string }>;
}

interface SidebarMenuGroup {
  key: string;
  label: string;
  items: MenuItem[];
  icon?: React.ComponentType<{ className?: string }>;
}

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  selectedModel?: string;
  onModelChange?: (model: string) => void;
}

// KPI Types
interface KpiOverallStats {
  total_tests: number;
  total_pass: number; 
  total_ng: number;
  accuracy_avg: number | null;
  current_queue: number;
}

type TempBand = 'AMBIENT' | 'HOT' | 'COLD';
type SpecBox = { target: string; spec: string; color: string };
type ModelSpec = Record<TempBand, SpecBox[]>;

const MODEL_OPTIONS = ['P703 DBL CAB', 'Test']; // เพิ่มโมเดลใหม่ได้ที่นี่

const MODEL_SPECS: Record<string, ModelSpec> = {
  'P703 DBL CAB': {
    AMBIENT: [
      { target: 'F#1', spec: '≤17', color: '#39c06b' },
      { target: 'F#2', spec: '≤20', color: '#39c06b' },
      { target: 'R#3', spec: '≤19', color: '#39c06b' },
    ],
    HOT: [
      { target: 'F#1', spec: '≤17', color: '#e23b2e' },
      { target: 'F#2', spec: '≤20', color: '#e23b2e' },
      { target: 'R#3', spec: '≤19', color: '#e23b2e' },
    ],
    COLD: [
      { target: 'F#1', spec: '≤21', color: '#3fa0f5' },
      { target: 'F#2', spec: '≤22', color: '#3fa0f5' },
      { target: 'R#3', spec: '≤20', color: '#3fa0f5' },
    ],
  },
  Test: {
    AMBIENT: [
      { target: 'A#1', spec: '≤15', color: '#39c06b' },
      { target: 'A#2', spec: '≤18', color: '#39c06b' },
      { target: 'B#3', spec: '≤16', color: '#39c06b' },
    ],
    HOT: [
      { target: 'A#1', spec: '≤15', color: '#e23b2e' },
      { target: 'A#2', spec: '≤18', color: '#e23b2e' },
      { target: 'B#3', spec: '≤16', color: '#e23b2e' },
    ],
    COLD: [
      { target: 'A#1', spec: '≤19', color: '#3fa0f5' },
      { target: 'A#2', spec: '≤20', color: '#3fa0f5' },
      { target: 'B#3', spec: '≤18', color: '#3fa0f5' },
    ],
  },
};

// KPI Helper Functions
function formatInt(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString() : "—";
}

function formatPercent(n: number | null | undefined): string {
  return n == null || Number.isNaN(n) ? "—" : `${n.toFixed(2)}%`;
}

async function getKpiOverallStats(model: string): Promise<KpiOverallStats> {
  // จำลอง API call - แทนที่ด้วย actual endpoint
  const response = await fetch(`/kpi/overall-stats?model=${model}`, {
    cache: 'no-store'
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch KPI stats: ${response.statusText}`);
  }
  
  return response.json();
}

// Sidebar Component
export const Sidebar: React.FC<SidebarProps> = ({ 
  isCollapsed, 
  onToggleCollapse,
  selectedModel = 'P703 DBL CAB',
  onModelChange
}) => {
  const pathname = usePathname();
  const [expandedMenus, setExpandedMenus] = React.useState<{[key: string]: boolean}>({});
  const [showScrollToTop, setShowScrollToTop] = React.useState(false);
  const [model, setModel] = React.useState<string>(selectedModel);
  
  // KPI State
  const [kpiData, setKpiData] = React.useState<KpiOverallStats | null>(null);
  const [kpiLoading, setKpiLoading] = React.useState(true);
  const [kpiError, setKpiError] = React.useState<string | null>(null);

  const lastScrollY = useRef(0)

  const COLLAPSED_WIDTH = 80
  const [expandedWidth, setExpandedWidth] = React.useState<number>(320)

  // const [selectOpen, setSelectOpen] = React.useState(false);

  const [modelDropdownOpen, setModelDropdownOpen] = React.useState(false);
  const modelSelectRef = useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modelSelectRef.current && !modelSelectRef.current.contains(event.target as Node)) {
        setModelDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle model change
  const handleModelChange = (newModel: string) => {
    setModel(newModel);
    if (onModelChange) {
      onModelChange(newModel);
    }
  };

  // Fetch KPI data when model changes
  React.useEffect(() => {
    if (isCollapsed) return; // Don't fetch when collapsed
    
    let mounted = true;

    async function fetchKpiData() {
      try {
        setKpiError(null);
        const data = await getKpiOverallStats(model);
        
        if (mounted) {
          setKpiData(data);
          setKpiLoading(false);
        }
      } catch (err) {
        if (mounted) {
          setKpiError(err instanceof Error ? err.message : 'Failed to fetch KPI data');
          setKpiLoading(false);
        }
      }
    }

    // Initial fetch
    fetchKpiData();

    // Set up polling interval (30 seconds)
    const intervalId = window.setInterval(fetchKpiData, 30000);

    return () => {
      mounted = false;
      window.clearInterval(intervalId);
    };
  }, [model, isCollapsed]);

  const getExpandedWidth = (vw: number) => {
    if (vw >= 1600) return 320
    if (vw >= 1440) return 300
    if (vw >= 1280) return 280
    if (vw >= 1024) return 260
    return 240
  }

  React.useEffect(() => {
    const apply = () => {
      const ew = getExpandedWidth(window.innerWidth)
      setExpandedWidth(ew)
      const real = (isCollapsed ? COLLAPSED_WIDTH : ew)
      document.documentElement.style.setProperty("--sidebar-width", `${real}px`)
    }

    apply()
    window.addEventListener("resize", apply, { passive: true })
    return () => window.removeEventListener("resize", apply)
  }, [isCollapsed])

  // Scroll to top functionality
  React.useEffect(() => {
    const handleScroll = () => {
      const y = window.pageYOffset || document.documentElement.scrollTop
      const prev = lastScrollY.current
      const scrollingDown = y > prev

      if (y <= 300) {
        setShowScrollToTop(false)
      } else if (scrollingDown) { 
        setShowScrollToTop(true) // >300 Show BTN
      } else {
        setShowScrollToTop(false)
      }

      lastScrollY.current = y
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };
  
  // Menu data with icons - แยก Home, Reports, และ Model เป็น menu แยกกัน
  const menuData: SidebarMenuGroup[] = [
    {
      key: 'home',
      label: 'Home',
      icon: HomeIcon,
      items: [
        { label: 'Home', href: '/', icon: HomeIcon }
      ]
    },
    {
      key: 'reports',
      label: 'Reports',
      icon: ChartBarIcon,
      items: [
        { label: 'Reports', href: '/reports', icon: ChartBarIcon }
      ]
    },
  ]

  const isMenuActive = (items: MenuItem[]) => {
    return items.some(item => pathname === item.href);
  };

  const toggleMenu = (menuKey: string) => {
    setExpandedMenus(prev => ({
      ...prev,
      [menuKey]: !prev[menuKey]
    }));
  };

  return (
    <>
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: isCollapsed ? COLLAPSED_WIDTH : expandedWidth }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="fixed left-0 top-0 h-full shadow-2xl z-40 flex flex-col border-r border-white/10"
        style={{
          background: '#164799',
        }}
      >
        {/* Header */}
        <div className="p-4 border-b border-white/10">
          <div className="flex items-center justify-between">
            {/* Wrapped logo and title in an anchor tag for homepage navigation */}
            <AnimatePresence>
              {!isCollapsed && (
                <motion.a
                  href="/"
                  className="flex items-center space-x-3 hover:scale-110"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                >
                  <Image
                    src="/autoliv_logo.png"
                    alt="Autoliv"
                    width={60}
                    height={24}
                    className="w-15 h-6 flex-shrink-0"
                  />
                </motion.a>
              )}
            </AnimatePresence>

            <div className="relative">
              <button
                onClick={onToggleCollapse}
                aria-label={isCollapsed ? "Open sidebar" : "Close sidebar"}
                className={`
                  group inline-flex items-center justify-center cursor-pointer
                  ${isCollapsed ? "h-10 w-10" : "h-9 w-9"}
                  rounded-md text-white/80 hover:text-white hover:bg-white/10
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/40
                `}
              >
                <span
                  className={`
                    inline-flex items-center justify-center
                    ${isCollapsed ? "translate-x-[2px]" : ""}  
                  `}
                >
                  {isCollapsed ? (
                    <>
                      <Image 
                        src="/autoliv_logo.png"
                        alt="logo"
                        width={60}
                        height={24}
                        className="h-5 w-10 opacity-100 group-hover:opacity-0 transition-opacity duration-150"
                        priority
                      />
                      <PanelLeftOpen className="h-6 w-6 absolute opacity-0 group-hover:opacity-100 transition-opacity duration-150" />
                    </>
                  ) : (
                    <PanelLeftClose className="h-5 w-5" />
                  )}
                </span>

                {/* Tooltip */}
                <div
                  role="tooltip"
                  className={`
                    pointer-events-none absolute z-50 whitespace-nowrap rounded-md
                    bg-gray-900/95 text-white text-[11px] font-medium px-2 py-1 shadow-lg
                    opacity-0 transition-all duration-150
                    group-hover:opacity-100 group-focus-visible:opacity-100
                    ${isCollapsed
                      ? "left-full top-1/2 -translate-y-1/2 ml-2"
                      : "left-1/2 top-full -translate-x-1/2 mt-2"}
                  `}
                >
                  {isCollapsed ? "Open sidebar" : "Close sidebar"}
                  <span 
                    className={`
                      absolute w-2 h-2 rotate-45 bg-gray-900/95
                      ${isCollapsed
                        ? "left-0 top-1/2 -translate-y-1/2 -ml-1"
                        : "-top-1 left-1/2 -translate-x-1/2"}
                      `}
                    />
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto no-scrollbar p-4">
          <nav className="space-y-2">
            {menuData.map((menu) => {
              const isExpanded = expandedMenus[menu.key];
              const isActive = isMenuActive(menu.items);
              const Icon = menu.icon;

              return (
                <div key={menu.key} className="space-y-1">
                  <Link
                    href={menu.items[0].href} // ใช้ href ของ item แรก
                    className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg transition-all duration-200 group relative cursor-pointer ${
                      isActive
                        ? 'bg-white/20 text-white shadow-lg'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    } ${isCollapsed ? 'justify-center' : ''}`}
                    title={isCollapsed ? menu.label : undefined}
                  >
                    {/* Icon remains in fixed position */}
                    {Icon && <Icon className="w-5 h-5 flex-shrink-0 absolute left-3 top-1/2 transform -translate-y-1/2" />}
                    
                    {/* Label and space for icon */}
                    <div className="flex items-center w-full">
                      <div className="w-5 h-5 flex-shrink-0 mr-3"></div>
                      <AnimatePresence>
                        {!isCollapsed && (
                          <motion.span
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.2 }}
                            className="font-medium text-sm whitespace-nowrap"
                          >
                            {menu.label}
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </div>
                  </Link>
                </div>
              );
            })}
          </nav>

          {/* Model Selector and SPEC Section */}
          <AnimatePresence>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                transition={{ duration: 0.3, delay: 0.1 }}
                className="mt-4 pt-4 border-t border-white/10"
              >
                {/* Model Selector */}
                <div className="px-3 mb-4">
                  <label className="block text-white font-bold text-xs uppercase tracking-wider mb-2">
                    Select Model
                  </label>

                  <div className="relative" ref={modelSelectRef}>
                    <div
                      className="relative w-full text-sm rounded-md bg-white/90 text-gray-900 px-3 py-2 pr-8
                                  focus-within:ring-2 focus-within:ring-white/40 transition-all duration-200
                                hover:bg-white cursor-pointer flex items-center"
                      onClick={() => setModelDropdownOpen(!modelDropdownOpen)}
                    >
                      <span className="block truncate">{model}</span>
                      <motion.span
                        aria-hidden
                        className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2"
                        animate={{ rotate: modelDropdownOpen ? 180 : 0 }}
                        initial={false}
                        transition={{ type: "tween", duration: 0.18, ease: "easeInOut" }}
                      >
                        <ChevronDown className="h-4 w-4 text-gray-700/80" />
                      </motion.span>
                    </div>

                    {/* Custom Dropdown Options */}
                    <div
                      className={`absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg transition-all duration-200 ${
                        modelDropdownOpen
                          ? "opacity-100 transform scale-100 translate-y-0"
                          : "opacity-0 transform scale-95 -translate-y-2 pointer-events-none"
                      }`}
                    >
                      <div className="py-1 max-h-60 overflow-auto">
                        {MODEL_OPTIONS.map((option) => (
                          <div
                            key={option}
                            className={`px-3 py-2 cursor-pointer hover:bg-blue-50 hover:text-blue-700 transition-colors duration-150 flex items-center justify-between ${
                              model === option
                                ? "bg-blue-100 text-blue-700 font-medium"
                                : "text-gray-900"
                            }`}
                            onClick={() => {
                              handleModelChange(option);
                              setModelDropdownOpen(false);
                            }}
                          >
                            <span className="block truncate">{option}</span>
                            {model === option && (
                              <Check className="w-4 h-4 text-blue-600" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* SPEC Title */}
                {/* <div className="px-3 mb-3 flex items-center justify-between">
                  <h3 className="text-white font-bold text-xs uppercase tracking-wider">
                    SPEC - {model}
                  </h3>
                </div> */}

                {/* SPEC Content */}
                {(() => {
                  const specs = MODEL_SPECS[model] ?? MODEL_SPECS['P703 DBL CAB']; // fallback
                  const renderBand = (title: TempBand) => (
                    <motion.div
                      key={`${model}-${title}`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="text-white/90 font-semibold mb-2 uppercase text-xs tracking-wide">{title}</div>
                      <div className="grid grid-cols-3 gap-2">
                        {specs[title].map((item, idx) => (
                          <motion.div
                            key={`${title}-${idx}`}
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ duration: 0.2, delay: idx * 0.05 }}
                            className="relative rounded-lg px-3 py-2 text-center ring-1 ring-white/20 shadow-[0_10px_18px_rgba(0,0,0,.35)] before:content-[''] before:absolute before:inset-x-0 before:top-0 before:h-1/2 before:rounded-t-lg before:bg-gradient-to-b before:from-white/30 before:to-transparent after:content-[''] after:absolute after:inset-x-0 after:bottom-0 after:h-1/3 after:rounded-b-lg after:bg-gradient-to-t after:from-black/20 after:to-transparent"
                            style={{ backgroundColor: item.color }}
                          >
                            <div className="text-white font-semibold text-sm">{item.target}</div>
                            <div className="text-white/90 text-xs">{item.spec}ms</div>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  );
                  return (
                    <div className="px-3 space-y-4">
                      {renderBand('AMBIENT')}
                      {renderBand('HOT')}
                      {renderBand('COLD')}
                    </div>
                  );
                })()}

                {/* KPI Stats Section */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.2 }}
                  className="mt-6 pt-4 border-t border-white/10"
                >
                  {/* <div className="px-3 mb-3">
                    <h3 className="text-white font-bold text-xs uppercase tracking-wider">
                      KPI - {model}
                    </h3>
                  </div> */}

                  {kpiLoading ? (
                    <div className="px-3 space-y-3">
                      {Array.from({ length: 3 }, (_, i) => (
                        <div key={i} className="animate-pulse">
                          <div className="bg-white/10 rounded-lg p-3">
                            <div className="h-4 bg-white/20 rounded mb-2"></div>
                            <div className="h-6 bg-white/20 rounded"></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : kpiError ? (
                    <div className="px-3">
                      <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-3">
                        <div className="text-red-300 text-xs">
                          Error loading KPI data
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="px-3 grid grid-cols-1 gap-3">
                      {[
                        {
                          label: "TOTAL IN SPEC",
                          value: formatInt(kpiData?.total_pass ?? 0),
                          bgColor: "bg-emerald-500/20",
                          borderColor: "border-emerald-500/30",
                          textColor: "text-emerald-300",
                          valueColor: "text-white"
                        },
                        {
                          label: "TOTAL OUT OF SPEC",
                          value: formatInt(kpiData?.total_ng ?? 0),
                          bgColor: "bg-rose-500/20",
                          borderColor: "border-rose-500/30",
                          textColor: "text-rose-300",
                          valueColor: "text-white"
                        },
                        {
                          label: "TOTAL RELIABILITY RATE",
                          value: formatPercent(kpiData?.accuracy_avg),
                          bgColor: "bg-violet-500/20",
                          borderColor: "border-violet-500/30",
                          textColor: "text-violet-300",
                          valueColor: "text-white"
                        }
                      ].map((item, idx) => (
                        <motion.div
                          key={item.label}
                          initial={{ scale: 0.95, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ duration: 0.2, delay: idx * 0.05 }}
                          className={`${item.bgColor} ${item.borderColor} border rounded-lg p-4`}
                        >
                          <div className={`${item.textColor} text-[10px] font-medium uppercase tracking-wider mb-2 text-center`}>
                            {item.label}
                          </div>
                          <div className={`${item.valueColor} font-bold text-3xl tabular-nums leading-none text-center`}>
                            {item.value}
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.aside>

      {/* Scroll to Top Button */}
      <AnimatePresence>
        {showScrollToTop && (
          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            onClick={scrollToTop}
            className="fixed bottom-6 left-1/2 -translate-x-1/2 z-30
                      bg-white px-2 py-1.5 rounded-full shadow-lg hover:shadow-xl
                      transition-all duration-300 hover:scale-105 backdrop-blur-sm border border-white/20
                      flex items-center space-x-2 font-semibold cursor-pointer"
            aria-label="Back to top"
          >
            <ArrowUpIcon className="w-4 h-4" style={{ color: 'var(--primary-color)' }} />
            <span className="text-sm" style={{ color: 'var(--primary-color)' }}>Back To Top</span>
          </motion.button>
        )}
      </AnimatePresence>
    </>
  );
};

// Hook for managing sidebar state
export const useSidebar = () => {
  const [isCollapsed, setIsCollapsed] = React.useState(false);
  const [selectedModel, setSelectedModel] = React.useState('P703 DBL CAB');

  const toggle = () => setIsCollapsed(!isCollapsed);
  const collapse = () => setIsCollapsed(true);
  const expand = () => setIsCollapsed(false);

  const handleModelChange = (model: string) => {
    setSelectedModel(model);
  };

  return { 
    isCollapsed, 
    toggle, 
    collapse, 
    expand,
    selectedModel,
    handleModelChange
  };
};