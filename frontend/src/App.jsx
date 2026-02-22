import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import './index.css';

function App() {
  const [activeView, setActiveView] = useState('dashboard');

  return (
    <div className="flex h-screen bg-slate-950">
      <Sidebar activeView={activeView} onViewChange={setActiveView} />
      <main className="flex-1 overflow-hidden">
        {activeView === 'dashboard' && <Dashboard />}
        {activeView === 'patients' && (
          <div className="flex items-center justify-center h-full text-slate-500">
            Patient management — integrated in Dashboard view
          </div>
        )}
        {activeView === 'settings' && (
          <div className="flex items-center justify-center h-full text-slate-500">
            <div className="text-center">
              <h2 className="text-lg font-bold text-white mb-2">Settings</h2>
              <p className="text-sm">
                Pain threshold: 4 (moderate) | 7 (severe)
                <br />
                Facial weight: 70% | Audio weight: 30%
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
