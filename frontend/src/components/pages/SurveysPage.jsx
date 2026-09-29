import React, { useState, useMemo } from 'react';
import { useToast } from '../../context/ToastContext';

export const SurveysPage = ({
  onNavigate,
  surveys = [],
  onCreateSurvey,
  currentSurveyId,
  onSelectSurvey,
}) => {
  const { showToast } = useToast();
  const [submitting, setSubmitting] = useState(false);

  // ---------------------------------------------------------
  // FORM DATA
  // Keep user-entered fields EMPTY.
  // Placeholder text is only a hint and is NOT a value.
  // ---------------------------------------------------------
  const [formData, setFormData] = useState({
    name: '',
    date: new Date().toISOString().slice(0, 10),
    waterBody: '',
    startLat: '',
    startLon: '',
    vesselName: '',
    depth: '',
    sonarDevice: 'Side-Scan Sonar (Dual Frequency 455/900 kHz)',
  });

  // Validation state
  const [errors, setErrors] = useState({});

  // ---------------------------------------------------------
  // ACTIVE SURVEY
  // ---------------------------------------------------------
  const activeSurvey = useMemo(() => {
    if (!currentSurveyId) {
      if (surveys.length > 0) {
        const sorted = [...surveys].sort(
          (a, b) => (Number(a.id) || 0) - (Number(b.id) || 0)
        );

        return sorted[0];
      }

      return null;
    }

    return (
      surveys.find(
        (s) => String(s.id) === String(currentSurveyId)
      ) || null
    );
  }, [surveys, currentSurveyId]);

  const activeSurveyCode = activeSurvey
    ? `SURV-${String(activeSurvey.id).padStart(3, '0')}`
    : null;

  // ---------------------------------------------------------
  // INPUT CHANGE
  // ---------------------------------------------------------
  const handleInputChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // Remove error immediately when user starts correcting it
    setErrors((prev) => {
      if (!prev[name]) return prev;

      const updated = { ...prev };
      delete updated[name];

      return updated;
    });
  };

  // ---------------------------------------------------------
  // VALIDATION
  // ---------------------------------------------------------
  const validateForm = () => {
    const newErrors = {};

    // Survey name
    if (!formData.name.trim()) {
      newErrors.name = 'Please enter a survey mission name.';
    } else if (formData.name.trim().length < 3) {
      newErrors.name = 'Survey name should contain at least 3 characters.';
    }

    // Water body
    if (!formData.waterBody.trim()) {
      newErrors.waterBody =
        'Please enter the water body or sea area.';
    }

    // Vessel
    if (!formData.vesselName.trim()) {
      newErrors.vesselName =
        'Please enter the survey vessel name.';
    }

    // Depth
    if (!formData.depth.trim()) {
      newErrors.depth = 'Please enter the survey depth.';
    } else if (!Number.isFinite(Number(formData.depth))) {
      newErrors.depth = 'Please enter a valid numeric depth.';
    } else if (Number(formData.depth) <= 0) {
      newErrors.depth = 'Depth must be greater than 0 meters.';
    }

    // Latitude
    if (!formData.startLat.trim()) {
      newErrors.startLat =
        'Please enter the starting latitude.';
    } else if (!Number.isFinite(Number(formData.startLat))) {
      newErrors.startLat =
        'Please enter a valid latitude.';
    } else if (
      Number(formData.startLat) < -90 ||
      Number(formData.startLat) > 90
    ) {
      newErrors.startLat =
        'Latitude must be between -90° and 90°.';
    }

    // Longitude
    if (!formData.startLon.trim()) {
      newErrors.startLon =
        'Please enter the starting longitude.';
    } else if (!Number.isFinite(Number(formData.startLon))) {
      newErrors.startLon =
        'Please enter a valid longitude.';
    } else if (
      Number(formData.startLon) < -180 ||
      Number(formData.startLon) > 180
    ) {
      newErrors.startLon =
        'Longitude must be between -180° and 180°.';
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  // ---------------------------------------------------------
  // FIELD ERROR COMPONENT
  // ---------------------------------------------------------
  const FieldError = ({ message }) => {
    if (!message) return null;

    return (
      <div className="mt-1.5 flex items-start gap-1.5 rounded-md border border-orange-200 bg-orange-50 px-2 py-1.5 text-[10.5px] leading-tight text-orange-700">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="w-3.5 h-3.5 flex-shrink-0 mt-[1px]"
        >
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
          <path d="M10.3 3.8 2.6 17.2a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 3.8a2 2 0 0 0-3.4 0Z" />
        </svg>

        <span>{message}</span>
      </div>
    );
  };

  // ---------------------------------------------------------
  // INPUT CLASS
  // ---------------------------------------------------------
  const getInputClass = (fieldName, extra = '') => {
    const hasError = Boolean(errors[fieldName]);

    return `
      w-full
      border
      rounded-lg
      px-3
      py-1.5
      text-[12px]
      font-medium
      text-slate-900
      bg-white
      placeholder:text-slate-300
      placeholder:font-normal
      focus:outline-none
      focus:ring-1
      transition-all
      ${
        hasError
          ? 'border-orange-300 focus:border-orange-400 focus:ring-orange-300/40 bg-orange-50/20'
          : 'border-slate-300 focus:border-teal-500 focus:ring-teal-500'
      }
      ${extra}
    `;
  };

  // ---------------------------------------------------------
  // CREATE SURVEY
  // ---------------------------------------------------------
  const handleCreate = async (e) => {
    e.preventDefault();

    // Run complete validation
    const isValid = validateForm();

    if (!isValid) {
      showToast({
        type: 'warning',
        message: 'Please complete the highlighted survey fields.',
      });

      return;
    }

    setSubmitting(true);

    const payload = {
      name: formData.name.trim(),
      date: formData.date,

      water_body: formData.waterBody.trim(),

      vessel: formData.vesselName.trim(),

      latitude:
        formData.startLat.trim() !== ''
          ? Number(formData.startLat)
          : null,

      longitude:
        formData.startLon.trim() !== ''
          ? Number(formData.startLon)
          : null,

      depth:
        formData.depth.trim() !== ''
          ? Number(formData.depth)
          : null,

      sonar_device: formData.sonarDevice,
    };

    try {
      const created = await onCreateSurvey(payload);

      const survCode = `SURV-${String(created.id).padStart(3, '0')}`;

      showToast({
        type: 'success',
        message: `Survey ${survCode} created successfully! Ready for Sonar Analysis.`,
      });

      if (onSelectSurvey) {
        onSelectSurvey(created.id);
      }

      // Clear validation errors after successful creation
      setErrors({});
    } catch (err) {
      showToast({
        type: 'error',
        message:
          err?.message || 'Failed to create survey.',
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="animate-page-fade space-y-6 bg-[#EFF8FB] min-h-screen -m-6 p-6">

      {/* =====================================================
          HEADER
      ===================================================== */}
      <div>
        <p className="text-[16px] text-slate-900 font-medium">
          Create or select a survey mission to generate a unique
          Survey ID before uploading sonar imagery.
        </p>
      </div>

      {/* =====================================================
          ACTIVE SURVEY BANNER
      ===================================================== */}
      {activeSurvey ? (
        <div className="bg-white border-2 border-teal-500 rounded-xl p-4 md:p-5 shadow-sm relative overflow-hidden">

          {/* Status badge */}
          <div className="absolute top-0 right-0 bg-teal-500 text-white font-mono text-[10px] font-bold px-3 py-1 rounded-bl-lg uppercase tracking-wider">
            Ready for Sonar Scans
          </div>

          {/* Survey Header */}
          <div className="flex items-center gap-2 pr-40 mb-4">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse flex-shrink-0" />

            <span className="font-mono text-[12px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-1 rounded-md whitespace-nowrap">
              Active Survey: {activeSurveyCode}
            </span>

            <span className="text-[14px] font-bold text-slate-800 truncate">
              {activeSurvey.name}
            </span>
          </div>

          {/* Survey Information */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">

            {/* Location */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5">
              <span className="text-[10px] text-slate-700 block font-bold uppercase tracking-wider mb-1">
                Water Body / Location
              </span>

              <span className="font-bold text-slate-900 text-[12px] truncate block">
                {activeSurvey.water_body ||
                  activeSurvey.location ||
                  'Coastal Zone'}
              </span>
            </div>

            {/* Vessel */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5">
              <span className="text-[10px] text-slate-700 block font-bold uppercase tracking-wider mb-1">
                Survey Vessel
              </span>

              <span className="font-bold text-slate-900 text-[12px] truncate block">
                {activeSurvey.vessel || 'Survey Vessel'}
              </span>
            </div>

            {/* Coordinates */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5">
              <span className="text-[10px] text-slate-700 block font-bold uppercase tracking-wider mb-1">
                Coordinates
              </span>

              <span className="font-bold text-slate-900 text-[12px] block truncate">
                {activeSurvey.latitude != null &&
                activeSurvey.longitude != null
                  ? `${Number(activeSurvey.latitude).toFixed(4)}, ${Number(
                      activeSurvey.longitude
                    ).toFixed(4)}`
                  : '—'}
              </span>
            </div>

            {/* Sonar */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5">
              <span className="text-[10px] text-slate-700 block font-bold uppercase tracking-wider mb-1">
                Sonar Device
              </span>

              <span className="font-bold text-slate-900 text-[12px] truncate block">
                {activeSurvey.sonar_device ||
                  'Side-Scan Sonar'}
              </span>
            </div>
          </div>

          {/* Bottom Actions */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-3 border-t border-slate-200">

            {/* Switch Survey */}
            {surveys.length > 1 ? (
              <div className="flex items-center gap-2">
                <span className="text-[10.5px] font-bold uppercase tracking-wider text-slate-700">
                  Switch Survey
                </span>

                <select
                  value={activeSurvey.id}
                  onChange={(e) =>
                    onSelectSurvey &&
                    onSelectSurvey(Number(e.target.value))
                  }
                  className="h-8 w-full max-w-[210px] border border-slate-200 bg-white text-slate-700 rounded-lg px-2.5 text-[11px] font-mono font-semibold focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 cursor-pointer"
                >
                  {surveys.map((s) => (
                    <option key={s.id} value={s.id}>
                      SURV-{String(s.id).padStart(3, '0')} ·{' '}
                      {s.name || 'Survey'}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div />
            )}

            {/* Proceed */}
            <button
              onClick={() => onNavigate('sonar')}
              className="bg-teal-600 hover:bg-teal-700 text-white font-semibold text-[12px] h-9 px-4 flex items-center justify-center gap-2 shadow-sm rounded-lg transition-colors"
            >
              <span>Proceed to Sonar Analysis</span>

              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                className="w-4 h-4"
              >
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-amber-800 text-[12px] flex items-center gap-3">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="w-5 h-5 text-amber-600 flex-shrink-0"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>

          <div>
            <b>No active survey exists yet.</b>{' '}
            Please create a new survey below. A unique Survey ID
            (e.g. <code>SURV-001</code>) will be generated to
            initialize sonar analysis.
          </div>
        </div>
      )}

      {/* =====================================================
          CREATE NEW SURVEY FORM
      ===================================================== */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">

        {/* Form Header */}
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-200">

          <div className="flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold text-xs border border-teal-200">
              +
            </span>

            <div>
              <h2 className="text-[14px] font-bold text-slate-900 leading-none">
                Create New Survey Mission
              </h2>

              <span className="text-[11.5px] text-slate-700 font-medium">
                Define the survey telemetry and metadata parameters
              </span>
            </div>
          </div>

          <span className="text-[10px] font-mono bg-teal-50 border border-teal-300 text-teal-800 px-2 py-0.5 rounded-full font-bold">
            Generates SURV-xxx ID
          </span>
        </div>

        <form onSubmit={handleCreate} noValidate>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">

            {/* =================================================
                SURVEY NAME
            ================================================= */}
            <div className="col-span-1 md:col-span-2">

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Survey Mission Name *
              </label>

              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                placeholder="e.g. Visakhapatnam Port Channel Sweep - Phase 2"
                className={getInputClass('name')}
              />

              <FieldError message={errors.name} />
            </div>

            {/* =================================================
                SURVEY DATE
            ================================================= */}
            <div>

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Survey Date
              </label>

              <input
                type="date"
                name="date"
                value={formData.date}
                onChange={handleInputChange}
                className={getInputClass(
                  'date',
                  'font-mono'
                )}
              />
            </div>

            {/* =================================================
                WATER BODY
            ================================================= */}
            <div>

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Water Body / Sea Area *
              </label>

              <input
                type="text"
                name="waterBody"
                value={formData.waterBody}
                onChange={handleInputChange}
                placeholder="Bay of Bengal / Gulf of Mannar"
                className={getInputClass('waterBody')}
              />

              <FieldError message={errors.waterBody} />
            </div>

            {/* =================================================
                VESSEL NAME
            ================================================= */}
            <div>

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Survey Vessel Name *
              </label>

              <input
                type="text"
                name="vesselName"
                value={formData.vesselName}
                onChange={handleInputChange}
                placeholder="INS Sagardhwani"
                className={getInputClass('vesselName')}
              />

              <FieldError message={errors.vesselName} />
            </div>

            {/* =================================================
                DEPTH
            ================================================= */}
            <div>

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Survey Depth (m) *
              </label>

              <input
                type="number"
                step="any"
                name="depth"
                value={formData.depth}
                onChange={handleInputChange}
                placeholder="42.5"
                className={getInputClass(
                  'depth',
                  'font-mono'
                )}
              />

              <FieldError message={errors.depth} />
            </div>

            {/* =================================================
                LATITUDE
            ================================================= */}
            <div>

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Start Latitude (°N) *
              </label>

              <input
                type="number"
                step="any"
                name="startLat"
                value={formData.startLat}
                onChange={handleInputChange}
                placeholder="17.6868"
                className={getInputClass(
                  'startLat',
                  'font-mono'
                )}
              />

              <FieldError message={errors.startLat} />
            </div>

            {/* =================================================
                LONGITUDE
            ================================================= */}
            <div>

              <label className="block text-[11px] font-mono text-slate-800 font-bold mb-1">
                Start Longitude (°E) *
              </label>

              <input
                type="number"
                step="any"
                name="startLon"
                value={formData.startLon}
                onChange={handleInputChange}
                placeholder="83.2185"
                className={getInputClass(
                  'startLon',
                  'font-mono'
                )}
              />

              <FieldError message={errors.startLon} />
            </div>
          </div>

          {/* =================================================
              FORM FOOTER
          ================================================= */}
          <div className="flex items-center justify-between pt-3 mt-4 border-t border-slate-200 flex-wrap gap-3">

            <div className="text-[11px] text-slate-700 font-medium">
              <span className="text-orange-600">*</span>{' '}
              Required fields must be completed before creating
              the survey.
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="btn primary bg-teal-600 hover:bg-teal-700 text-white font-medium text-[12.5px] py-2 px-5 rounded-lg flex items-center gap-2 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? (
                <>
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="w-4 h-4 animate-spin"
                  >
                    <polyline points="23 4 23 10 17 10" />
                    <polyline points="1 20 1 14 7 14" />
                  </svg>

                  <span>
                    Generating Survey ID...
                  </span>
                </>
              ) : (
                <>
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    className="w-4 h-4"
                  >
                    <line
                      x1="12"
                      y1="5"
                      x2="12"
                      y2="19"
                    />

                    <line
                      x1="5"
                      y1="12"
                      x2="19"
                      y2="12"
                    />
                  </svg>

                  <span>
                    Create Survey & Generate ID
                  </span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* =====================================================
          HISTORICAL RECORDS
      ===================================================== */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex items-center justify-between gap-4 text-xs text-slate-600">

        <div className="flex items-center gap-2.5">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="w-4 h-4 text-teal-600 flex-shrink-0"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>

          <span>
            <b>Historical Mission Logs:</b>{' '}
            Comprehensive records of past surveys and telemetry
            exports are archived under the{' '}
            <b>Reports</b> section.
          </span>
        </div>

        <button
          onClick={() => onNavigate('reports')}
          className="btn secondary text-[11px] py-1 px-3 whitespace-nowrap"
        >
          Open Reports →
        </button>
      </div>
    </section>
  );
};

export default SurveysPage;