import React, { useState } from 'react';
import { useToast } from '../../context/ToastContext';
import { Switch } from '../common/Switch';
import { defaultSystemSettings } from '../../data/mockData';

export const SettingsPage = () => {
  const { showToast } = useToast();
  const [settings, setSettings] = useState(defaultSystemSettings);

  const handleToggle = (key, label) => {
    const nextState = !settings.preprocessing[key];
    setSettings((prev) => ({
      ...prev,
      preprocessing: {
        ...prev.preprocessing,
        [key]: nextState,
      },
    }));
    showToast({
      type: 'info',
      message: `${label} ${nextState ? 'enabled' : 'disabled'}.`,
    });
  };

  const handleConfidenceChange = (e) => {
    const val = parseFloat(e.target.value);
    setSettings((prev) => ({
      ...prev,
      confidenceThreshold: val,
    }));
  };

  return (
    <section className="animate-page-fade space-y-6 bg-[#EFF8FB] min-h-screen -m-6 p-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* AI Model Status */}
        <div className="bg-navy-850 border border-navy-800 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-4 md:p-5 flex flex-col justify-between">
          <div>
            <div className="text-[12.5px] font-semibold text-foam-50 mb-3.5">AI model status & architecture</div>
            <div className="space-y-2 text-[12.5px]">
              <div className="flex justify-between py-2 border-b border-dashed border-navy-800">
                <span className="text-slate-400">YOLO model</span>
                <span className="font-mono text-sea-400">● Online</span>
              </div>
              <div className="flex justify-between py-2 border-b border-dashed border-navy-800">
                <span className="text-slate-400">Model checkpoint</span>
                <span className="font-mono font-medium text-foam-50">{settings.modelCheckpoint}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-dashed border-navy-800">
                <span className="text-slate-400">Training curriculum</span>
                <span className="font-mono font-medium text-foam-50">{settings.trainingCurriculum}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-dashed border-navy-800">
                <span className="text-slate-400">Architecture</span>
                <span className="font-mono font-medium text-foam-50">{settings.architecture}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-dashed border-navy-800">
                <span className="text-slate-400">Input resolution</span>
                <span className="font-mono font-medium text-foam-50">{settings.inputResolution}</span>
              </div>
              <div className="flex justify-between py-2 border-b border-dashed border-navy-800">
                <span className="text-slate-400">Inference device</span>
                <span className="font-mono font-medium text-foam-50">{settings.inferenceDevice}</span>
              </div>
            </div>
          </div>

          <div className="mt-4">
            <label className="block text-[10.5px] text-slate-400 mb-1 font-mono">
              Global confidence baseline threshold ({settings.confidenceThreshold.toFixed(2)})
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={settings.confidenceThreshold}
              onChange={handleConfidenceChange}
              className="w-full accent-teal-400 cursor-pointer"
            />
          </div>
        </div>

        {/* Preprocessing Toggles */}
        <div className="bg-navy-850 border border-navy-800 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-4 md:p-5 flex flex-col justify-between">
          <div>
            <div className="text-[12.5px] font-semibold text-foam-50 mb-3.5">Physics-constrained preprocessing</div>
            <div className="space-y-1 text-[12.5px]">
              <div className="flex justify-between items-center py-2.5 border-b border-navy-800">
                <div>
                  <div className="text-foam-50 font-medium">Cross-track swath correction</div>
                  <div className="text-[11px] text-slate-400">Range-dependent acoustic attenuation compensation</div>
                </div>
                <Switch
                  checked={settings.preprocessing.swathCorrection}
                  onChange={() => handleToggle('swathCorrection', 'Swath correction')}
                />
              </div>
              <div className="flex justify-between items-center py-2.5 border-b border-navy-800">
                <div>
                  <div className="text-foam-50 font-medium">Bilateral speckle denoising</div>
                  <div className="text-[11px] text-slate-400">Edge-preserving acoustic speckle noise suppression</div>
                </div>
                <Switch
                  checked={settings.preprocessing.denoising}
                  onChange={() => handleToggle('denoising', 'Denoising')}
                />
              </div>
              <div className="flex justify-between items-center py-2.5 border-b border-navy-800">
                <div>
                  <div className="text-foam-50 font-medium">Percentile normalization</div>
                  <div className="text-[11px] text-slate-400">Robust 1%–99% acoustic intensity range re-scaling</div>
                </div>
                <Switch
                  checked={settings.preprocessing.normalization}
                  onChange={() => handleToggle('normalization', 'Normalization')}
                />
              </div>
              <div className="flex justify-between items-center py-2.5">
                <div>
                  <div className="text-foam-50 font-medium">CIELAB CLAHE contrast</div>
                  <div className="text-[11px] text-slate-400">L-channel adaptive histogram equalization</div>
                </div>
                <Switch
                  checked={settings.preprocessing.clahe}
                  onChange={() => handleToggle('clahe', 'CLAHE contrast')}
                />
              </div>
            </div>
          </div>
          <div className="mt-3 text-[11px] font-mono text-slate-400 bg-navy-900/60 p-2.5 rounded-lg border border-navy-800">
            Pipeline: Swath Norm → Bilateral Filter → Percentile Scale → CLAHE → YOLO Inference
          </div>
        </div>
      </div>

      {/* Target Classes & Calibrated Physics Thresholds */}
      <div>
        <h2 className="text-[14px] font-semibold text-foam-50 mb-3.5 flex items-center gap-2">
          <span>Target classes & calibrated physics thresholds</span>
          <span className="text-[10.5px] font-mono text-slate-500 font-normal">
            class-specific acoustic shadow & highlight tuning
          </span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
          {settings.classes.map((cls) => (
            <div
              key={cls.id}
              className="bg-navy-850 border border-navy-800 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-4 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-foam-50 text-[13px] flex items-center gap-1.5">
                    <span>{cls.icon}</span>
                    <span>{cls.name}</span>
                  </span>
                  <span className="font-mono text-[11px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded">
                    Thresh: {cls.threshold.toFixed(2)}
                  </span>
                </div>
                <p className="text-[11.5px] text-slate-400 leading-relaxed">
                  {cls.rationale}
                </p>
              </div>
              <div className="mt-3 pt-2 border-t border-dashed border-navy-800 flex justify-between items-center text-[10.5px] font-mono text-slate-500">
                <span>Class ID: {cls.id}</span>
                <span className="text-teal-600 font-medium">Calibrated</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Risk Score Weighting */}
      <div>
        <h2 className="text-[14px] font-semibold text-foam-50 mb-3.5 flex items-center gap-2">
          <span>Risk score weighting</span>
          <span className="text-[10.5px] font-mono text-slate-500 font-normal">
            acoustic telemetry calibration weights (risk_engine.py)
          </span>
        </h2>
        <div className="bg-navy-850 border border-navy-800 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-300 p-4 md:p-5 space-y-3">
          {settings.weights.map((w) => (
            <div key={w.label} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400 font-medium">{w.label}</span>
                <span className="font-mono text-foam-50 font-bold">{w.percent}%</span>
              </div>
              <div className="h-1.5 bg-navy-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-teal-400 rounded-full transition-all duration-300"
                  style={{ width: `${w.percent}%` }}
                ></div>
              </div>
              {w.detail && (
                <div className="text-[10.5px] text-slate-400 font-mono">{w.detail}</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};