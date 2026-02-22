import { useState } from 'react';
import { FiActivity, FiUsers, FiSettings, FiMenu, FiX, FiHeart } from 'react-icons/fi';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: FiActivity },
  { id: 'patients', label: 'Patients', icon: FiUsers },
  { id: 'settings', label: 'Settings', icon: FiSettings },
];

export default function Sidebar({ activeView, onViewChange }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`h-screen bg-slate-900 border-r border-slate-700 flex flex-col transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-56'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <FiHeart className="text-cyan-400 text-xl" />
            <span className="font-bold text-lg text-cyan-400">NeoGuard</span>
          </div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-slate-400 hover:text-white p-1"
        >
          {collapsed ? <FiMenu size={20} /> : <FiX size={20} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 text-sm transition-colors ${
                isActive
                  ? 'bg-cyan-500/10 text-cyan-400 border-r-2 border-cyan-400'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Icon size={20} />
              {!collapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
          NeoGuard v1.0
          <br />
          NICU Pain Monitor
        </div>
      )}
    </aside>
  );
}
