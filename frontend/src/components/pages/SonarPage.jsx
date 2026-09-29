import React, { useRef, useState } from 'react';
import { useToast } from '../../context/ToastContext';
import { api } from '../../services/api';

import {
  normalizeDetection,
  calculateOverallRisk,
  getConfidenceCategory,
  getRiskBadgeStyle,
  formatConfidence,
  formatCoordinate,
} from '../../utils/detectionUtils';

// Backend hard floor
const INTERNAL_INFERENCE_THRESHOLD = 0.25;

// ---------------------------------------------------------
// 7-STAGE PIPELINE
// ---------------------------------------------------------
const PREPROCESSING_STAGES = [
  'Raw Sonar',
  'Preprocess',
  'Denoise',
  'Contrast',
  'Normalize',
  'YOLO AI',
  'Detection',
];

// ---------------------------------------------------------
// VALIDATION ERROR
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
const inputClass = (hasError = false) => `
  w-full
  border
  rounded-md
  px-2.5
  py-1.5
  text-[11px]
  font-mono
  font-semibold
  text-slate-900
  bg-white
  placeholder:text-slate-300
  placeholder:font-normal
  focus:outline-none
  focus:ring-1
  transition-all
  ${
    hasError
      ? 'border-orange-300 bg-orange-50/20 focus:border-orange-400 focus:ring-orange-300/40'
      : 'border-slate-300 focus:border-teal-500 focus:ring-teal-500'
  }
`;

// ---------------------------------------------------------
// COMPONENT
// ---------------------------------------------------------
export const SonarPage = ({
  onNavigate,
  onAnalysisComplete,
  onDetectionReviewed,
  initialSurveyId,
  activeSurvey,
  surveys = [],
  onSelectSurvey,
}) => {
  const { showToast } = useToast();
  const fileInputRef = useRef(null);

  // ---------------------------------------------------------
  // SURVEY RESOLUTION
  // ---------------------------------------------------------
  const effectiveSurvey =
    activeSurvey ||
    surveys.find(
      (s) => String(s.id) === String(initialSurveyId)
    ) ||
    (surveys.length > 0 ? surveys[0] : null);

  const hasActiveSurvey = Boolean(
    effectiveSurvey && effectiveSurvey.id
  );

  const activeSurveyCode = hasActiveSurvey
    ? `SURV-${String(effectiveSurvey.id).padStart(3, '0')}`
    : null;

  // ---------------------------------------------------------
  // FILE STATE
  // ---------------------------------------------------------
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedImageSrc, setUploadedImageSrc] = useState(null);
  const [processedImageSrc, setProcessedImageSrc] = useState(null);
  const [detectionImageSrc, setDetectionImageSrc] = useState(null);
  const [detections, setDetections] = useState([]);

  const [rawTag, setRawTag] = useState('NO SCAN LOADED');

  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoadingFile, setIsLoadingFile] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activePipelineStage, setActivePipelineStage] =
    useState(-1);

  // ---------------------------------------------------------
  // FORM STATE
  // ---------------------------------------------------------
  const [surveyId, setSurveyId] = useState(
    effectiveSurvey
      ? String(effectiveSurvey.id)
      : initialSurveyId
      ? String(initialSurveyId)
      : ''
  );

  const [depth, setDepth] = useState('');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');

  const [heading, setHeading] = useState('');
  const [timestamp, setTimestamp] = useState('');

  // ---------------------------------------------------------
  // VALIDATION
  // ---------------------------------------------------------
  const [errors, setErrors] = useState({});

  // ---------------------------------------------------------
  // STATUS
  // ---------------------------------------------------------
  const [statusMessage, setStatusMessage] = useState(
    'Upload a side-scan sonar image to begin analysis.'
  );

  // ---------------------------------------------------------
  // SYNC SURVEY
  // ---------------------------------------------------------
  React.useEffect(() => {
    if (effectiveSurvey) {
      setSurveyId(String(effectiveSurvey.id));
    }
  }, [effectiveSurvey]);

  // ---------------------------------------------------------
  // VALIDATE SONAR METADATA
  // ---------------------------------------------------------
  const validateMetadata = () => {
    const newErrors = {};

    if (!surveyId.trim()) {
      newErrors.surveyId =
        'Survey ID is required.';
    }

    if (!depth.trim()) {
      newErrors.depth =
        'Please enter the survey depth.';
    } else if (!Number.isFinite(Number(depth))) {
      newErrors.depth =
        'Please enter a valid numeric depth.';
    } else if (Number(depth) <= 0) {
      newErrors.depth =
        'Depth must be greater than 0 meters.';
    }

    if (!latitude.trim()) {
      newErrors.latitude =
        'Please enter the latitude.';
    } else if (!Number.isFinite(Number(latitude))) {
      newErrors.latitude =
        'Please enter a valid latitude.';
    } else if (
      Number(latitude) < -90 ||
      Number(latitude) > 90
    ) {
      newErrors.latitude =
        'Latitude must be between -90 and 90.';
    }

    if (!longitude.trim()) {
      newErrors.longitude =
        'Please enter the longitude.';
    } else if (!Number.isFinite(Number(longitude))) {
      newErrors.longitude =
        'Please enter a valid longitude.';
    } else if (
      Number(longitude) < -180 ||
      Number(longitude) > 180
    ) {
      newErrors.longitude =
        'Longitude must be between -180 and 180.';
    }

    if (heading.trim() && !Number.isFinite(Number(heading))) {
      newErrors.heading =
        'Please enter a valid heading.';
    } else if (
      heading.trim() &&
      (Number(heading) < 0 || Number(heading) > 360)
    ) {
      newErrors.heading =
        'Heading must be between 0° and 360°.';
    }

    setErrors(newErrors);

    return Object.keys(newErrors).length === 0;
  };

  // ---------------------------------------------------------
  // CLEAR FIELD ERROR
  // ---------------------------------------------------------
  const clearFieldError = (field) => {
    setErrors((prev) => {
      if (!prev[field]) return prev;

      const updated = { ...prev };
      delete updated[field];

      return updated;
    });
  };

  // ---------------------------------------------------------
  // PREVIEW FILE
  // ---------------------------------------------------------
  const previewFile = (file) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      setUploadedImageSrc(e.target.result);
    };

    reader.onerror = () => {
      showToast({
        type: 'error',
        message: 'Failed to preview sonar image.',
      });
    };

    reader.readAsDataURL(file);
  };

  // ---------------------------------------------------------
  // FILE PROCESS
  // ---------------------------------------------------------
  const handleFileProcess = (file) => {
    if (!hasActiveSurvey) {
      showToast({
        type: 'warning',
        message:
          'Create or select a survey before uploading sonar imagery.',
      });
      return;
    }

    const ext = file.name
      .split('.')
      .pop()
      .toLowerCase();

    const validExts = [
      'png',
      'jpg',
      'jpeg',
      'tiff',
      'tif',
    ];

    if (!validExts.includes(ext)) {
      showToast({
        type: 'error',
        message:
          'Invalid file format. Please upload PNG, JPG, or TIFF sonar scans.',
      });
      return;
    }

    setIsLoadingFile(true);

    setSelectedFile(file);
    setRawTag(file.name);

    const generatedTimestamp = new Date()
      .toISOString()
      .replace('T', ' ')
      .replace('Z', ' UTC');

    setTimestamp(generatedTimestamp);

    setDetections([]);
    setProcessedImageSrc(null);
    setDetectionImageSrc(null);

    setActivePipelineStage(0);

    setStatusMessage(
      `Scan loaded: "${file.name}". Ready to run 7-stage detection pipeline.`
    );

    previewFile(file);

    setIsLoadingFile(false);

    showToast({
      type: 'success',
      message: `Sonar scan "${file.name}" loaded for ${activeSurveyCode}. Click Analyze to run inference.`,
    });
  };

  // ---------------------------------------------------------
  // DRAG EVENTS
  // ---------------------------------------------------------
  const handleDragOver = (e) => {
    e.preventDefault();

    if (!hasActiveSurvey) return;

    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);

    if (!hasActiveSurvey) {
      showToast({
        type: 'warning',
        message:
          'Create or select a survey before uploading sonar imagery.',
      });
      return;
    }

    if (e.dataTransfer?.files?.[0]) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (!hasActiveSurvey) {
      showToast({
        type: 'warning',
        message:
          'Create or select a survey before uploading sonar imagery.',
      });
      return;
    }

    if (e.target?.files?.[0]) {
      handleFileProcess(e.target.files[0]);
    }
  };

  // ---------------------------------------------------------
  // ANALYZE FILE
  // ---------------------------------------------------------
  const analyzeFile = async (file = selectedFile) => {
    if (!hasActiveSurvey) {
      showToast({
        type: 'warning',
        message:
          'Create or select a survey before uploading sonar imagery.',
      });
      return;
    }

    if (!file) {
      showToast({
        type: 'warning',
        message:
          'Upload a sonar image first.',
      });
      return;
    }

    const metadataValid = validateMetadata();

    if (!metadataValid) {
      showToast({
        type: 'warning',
        message:
          'Please complete the highlighted survey fields before analysis.',
      });

      return;
    }

    setIsAnalyzing(true);

    setDetections([]);
    setProcessedImageSrc(null);
    setDetectionImageSrc(null);

    let currentStage = 0;

    setActivePipelineStage(0);
    setStatusMessage(
      `Pipeline: ${PREPROCESSING_STAGES[0]}...`
    );

    const stageInterval = setInterval(() => {
      currentStage += 1;

      if (
        currentStage <
        PREPROCESSING_STAGES.length - 1
      ) {
        setActivePipelineStage(currentStage);

        setStatusMessage(
          `Pipeline: ${PREPROCESSING_STAGES[currentStage]}...`
        );
      }
    }, 180);

    const targetSurveyId =
      effectiveSurvey?.id ??
      (surveyId ? Number(surveyId) : null);

    const surveyMeta = {
      surveyId: targetSurveyId,

      latitude:
        latitude === ''
          ? null
          : Number(latitude),

      longitude:
        longitude === ''
          ? null
          : Number(longitude),

      depth:
        depth === ''
          ? null
          : Number(depth),

      heading:
        heading === ''
          ? null
          : Number(heading),

      timestamp: timestamp || null,
    };

    try {
      const result =
        await api.analyzeSonarScan(file, {
          ...surveyMeta,
          confidenceThreshold:
            INTERNAL_INFERENCE_THRESHOLD,
        });

      clearInterval(stageInterval);

      setActivePipelineStage(
        PREPROCESSING_STAGES.length - 1
      );

      const effectiveSurveyMeta = {
        survey_id:
          result.survey_id ??
          targetSurveyId,

        survey_code:
          result.survey_code ??
          activeSurveyCode,

        latitude:
          result.survey?.latitude ??
          surveyMeta.latitude,

        longitude:
          result.survey?.longitude ??
          surveyMeta.longitude,

        depth:
          result.survey?.depth ??
          surveyMeta.depth,
      };

      const normalized =
        (result.detections || []).map((d) =>
          normalizeDetection(
            d,
            effectiveSurveyMeta
          )
        );

      setDetections(normalized);

      setProcessedImageSrc(
        result.processed_image || null
      );

      setDetectionImageSrc(
        result.detection_image || null
      );

      if (result.survey_id) {
        setSurveyId(
          String(result.survey_id)
        );
      }

      if (result.survey?.latitude != null) {
        setLatitude(
          String(result.survey.latitude)
        );
      }

      if (result.survey?.longitude != null) {
        setLongitude(
          String(result.survey.longitude)
        );
      }

      if (result.survey?.depth != null) {
        setDepth(
          String(result.survey.depth)
        );
      }

      const riskCalc =
        calculateOverallRisk(normalized);

      setStatusMessage(
        `Pipeline complete: ${normalized.length} detection${
          normalized.length === 1 ? '' : 's'
        } identified. Overall Risk: ${
          riskCalc.overallRiskLevel
        } (${riskCalc.overallRiskScore}/100).`
      );

      onAnalysisComplete?.(
        {
          ...result,

          detections: normalized,

          detectionImage:
            result.detection_image,

          processedImage:
            result.processed_image,

          uploadedImage:
            uploadedImageSrc,

          survey: {
            ...result.survey,

            heading:
              heading || null,

            timestamp:
              timestamp || null,
          },
        },
        normalized
      );

      showToast({
        type: normalized.length
          ? 'success'
          : 'info',

        message: normalized.length
          ? `Analysis complete: ${normalized.length} detection${
              normalized.length === 1 ? '' : 's'
            } found. Sent to review queue.`
          : 'Analysis complete: No objects detected in this scan.',
      });
    } catch (error) {
      clearInterval(stageInterval);

      setActivePipelineStage(-1);

      setStatusMessage(
        'Pipeline analysis failed. Check backend connection.'
      );

      showToast({
        type: 'error',
        message:
          error.message ||
          'Sonar analysis failed.',
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ---------------------------------------------------------
  // REVIEW
  // ---------------------------------------------------------
  const handleOpenReview = (detection) => {
    onNavigate('review', {
      selectedDetectionId: detection.id,
      detection,
    });
  };

  // ---------------------------------------------------------
  // QUICK REVIEW
  // ---------------------------------------------------------
  const handleReviewQuick = async (
    detection,
    verdict
  ) => {
    try {
      const response =
        await api.submitDetectionReview(
          detection.id,
          {
            verdict,
            reviewer: 'sonar-operator',

            comment:
              verdict === 'confirmed'
                ? 'Accepted detection'
                : verdict === 'flagged'
                ? 'Flagged for inspection'
                : 'Rejected false acoustic reflection',
          }
        );

      const updatedStatus =
        response?.new_status ||
        response?.status ||
        (
          verdict === 'confirmed'
            ? 'confirmed'
            : verdict === 'flagged'
            ? 'flagged'
            : 'rejected'
        );

      setDetections((prev) =>
        prev.map((item) =>
          item.id === detection.id
            ? {
                ...item,
                status: updatedStatus,
              }
            : item
        )
      );

      onDetectionReviewed?.(
        detection.id,
        updatedStatus
      );

      showToast({
        type:
          verdict === 'confirmed'
            ? 'success'
            : verdict === 'flagged'
            ? 'info'
            : 'warning',

        message: `${detection.detection_id} marked as ${updatedStatus}.`,
      });
    } catch (error) {
      showToast({
        type: 'error',
        message:
          error.message ||
          'Could not save review action.',
      });
    }
  };

  // ---------------------------------------------------------
  // RISK
  // ---------------------------------------------------------
  const overallRisk =
    calculateOverallRisk(detections);

  const overallRiskBadge =
    getRiskBadgeStyle(
      overallRisk.overallRiskLevel
    );

  const canAnalyze =
    !isAnalyzing &&
    !!selectedFile &&
    hasActiveSurvey;

  // =========================================================
  // RETURN
  // =========================================================
  return (
    <section className="animate-page-fade space-y-6 bg-[#EFF8FB] min-h-screen -m-6 p-6">

      {/* HEADER */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-1 border-b border-slate-100">
        <div>
          <h1 className="text-[18px] font-bold text-slate-800 leading-tight">
            Upload → Scan → Detect
          </h1>
        </div>

        <button
          onClick={() => onNavigate('review')}
          className={`btn text-[12px] py-1.5 px-3 ${
            detections.length > 0
              ? 'primary bg-teal-600 hover:bg-teal-700 text-white font-medium'
              : 'secondary'
          }`}
        >
          Human Verification{' '}
          {detections.length > 0
            ? `(${detections.length})`
            : ''}{' '}
          →
        </button>
      </div>

      {/* ACTIVE SURVEY */}
      {hasActiveSurvey ? (
        <div className="bg-white border-2 border-teal-500 rounded-xl p-3 shadow-sm flex flex-wrap items-center justify-between gap-3 sonar-survey-bar">

          <div className="flex flex-wrap items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />

            <span className="font-mono text-[12.5px] font-bold text-teal-700 bg-teal-50 border border-teal-200 px-2.5 py-0.5 rounded-md">
              Active Survey: {activeSurveyCode}
            </span>

            <span className="text-[13px] font-bold text-slate-800">
              {effectiveSurvey.name}
            </span>

            <span className="hidden sm:inline text-xs text-slate-500 font-mono">
              ·{' '}
              {effectiveSurvey.water_body ||
                effectiveSurvey.location ||
                'Ocean Zone'}{' '}
              ·{' '}
              {effectiveSurvey.vessel ||
                'Survey Vessel'}
            </span>
          </div>

          <div className="flex items-center gap-2 flex-wrap sonar-survey-bar-actions">
            {surveys.length > 1 && (
              <div className="flex items-center gap-1.5 text-xs">
                <span className="text-slate-400 font-mono text-[11px]">
                  Switch:
                </span>

                <select
                  value={effectiveSurvey.id}
                  onChange={(e) =>
                    onSelectSurvey &&
                    onSelectSurvey(
                      Number(e.target.value)
                    )
                  }
                  className="border border-slate-200 bg-slate-50 text-slate-700 rounded px-2 py-1 text-[11px] font-mono focus:outline-none focus:ring-1 focus:ring-teal-400 max-w-[160px] sm:max-w-none"
                >
                  {surveys.map((s, index) => (
                    <option
                      key={s.id ?? `survey-${index}`}
                      value={s.id}
                    >
                      SURV-
                      {String(s.id).padStart(
                        3,
                        '0'
                      )}{' '}
                      · {s.name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              onClick={() =>
                onNavigate('surveys')
              }
              className="btn secondary text-[11px] py-1 px-2.5 text-slate-600 hover:text-teal-700 font-medium whitespace-nowrap"
            >
              Survey Settings
            </button>
          </div>
        </div>

      ) : (
        <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-4 text-amber-900 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">

          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-amber-100 border border-amber-300 text-amber-800 flex items-center justify-center font-bold text-base flex-shrink-0">
              !
            </div>

            <div>
              <div className="text-[13px] font-bold">
                Survey Required Before Sonar Upload
              </div>

              <div className="text-[12px] text-amber-800">
                Create or select a survey before
                uploading sonar imagery.
              </div>
            </div>
          </div>

          <button
            onClick={() =>
              onNavigate('surveys')
            }
            className="btn primary bg-amber-600 hover:bg-amber-700 text-white font-medium text-[12px] py-1.5 px-3 flex-shrink-0 shadow-sm"
          >
            Create or Select Survey →
          </button>
        </div>
      )}

      {/* LOAD SONAR SCAN */}
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">

        <div className="flex items-center justify-between px-4 pt-3.5 pb-2.5 border-b border-slate-100">
          <div>
            <h2 className="text-[14px] font-bold text-slate-800">
              Load a Sonar Scan
            </h2>

            <p className="text-[12px] text-slate-400 mt-0.5">
              Upload an image, confirm the survey parameters,
              then run detection.
            </p>
          </div>

          {hasActiveSurvey && (
            <span className="font-mono text-[10.5px] text-teal-700 bg-teal-50 border border-teal-200 px-2 py-0.5 rounded-md flex-shrink-0">
              {activeSurveyCode}
            </span>
          )}
        </div>

        <div className="p-4 grid grid-cols-1 lg:grid-cols-[1fr_1.35fr] gap-10">

          {/* STEP 1 */}
          <div>
            <div className="text-[13px] font-semibold text-slate-800 mb-1.5">
              1. Sonar image
            </div>

            {!hasActiveSurvey ? (
              <div
                onClick={() => {
                  showToast({
                    type: 'warning',
                    message:
                      'Create or select a survey before uploading sonar imagery.',
                  });

                  onNavigate('surveys');
                }}
                className="bg-amber-50/60 border-2 border-dashed border-amber-300 rounded-xl h-[105px] p-3 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-amber-50 transition-colors"
              >
                <div className="w-8 h-8 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center mb-1.5">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="w-4 h-4"
                  >
                    <rect
                      x="3"
                      y="11"
                      width="18"
                      height="11"
                      rx="2"
                    />

                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                  </svg>
                </div>

                <div className="text-[12px] font-bold text-amber-900">
                  Upload disabled
                </div>

                <div className="text-[10px] text-amber-700 mt-0.5">
                  Select a survey first to enable uploads.
                </div>
              </div>
            ) : (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl h-[120px] px-3 py-2.5 flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${
                  isDragOver
                    ? 'border-teal-400 bg-teal-50/60'
                    : selectedFile
                    ? 'border-teal-200 bg-teal-50/30 hover:border-teal-300'
                    : 'border-slate-250 bg-slate-50/60 hover:border-teal-300 hover:bg-teal-50/30'
                }`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*,.tiff,.tif"
                  className="hidden"
                  onChange={handleFileInputChange}
                />

                {isLoadingFile ? (
                  <div className="flex items-center gap-1.5 text-xs text-teal-600">
                    <span className="w-3.5 h-3.5 border-2 border-teal-400/30 border-t-teal-500 rounded-full animate-spin" />

                    <span>Loading scan...</span>
                  </div>
                ) : (
                  <>
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center mb-1.5 ${
                        selectedFile
                          ? 'bg-teal-100 text-teal-700'
                          : 'bg-teal-50 text-teal-600'
                      }`}
                    >
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        className="w-4 h-4"
                      >
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />

                        <polyline points="17 8 12 3 7 8" />

                        <line
                          x1="12"
                          y1="3"
                          x2="12"
                          y2="15"
                        />
                      </svg>
                    </div>

                    <div className="text-[11.5px] font-semibold text-slate-700 truncate max-w-[220px]">
                      {selectedFile
                        ? rawTag
                        : 'Drop sonar image, or click to browse'}
                    </div>

                    <div className="text-[9.5px] text-slate-400 font-mono mt-0.5">
                      {selectedFile
                        ? 'Click or drop to replace'
                        : 'Supports PNG, JPG, TIFF'}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* STEP 2 */}
          <div>
            <div className="text-[13px] font-semibold text-slate-800 mb-1.5">
              2. Survey metadata
            </div>

            <div className="grid grid-cols-2 gap-2.5 text-left">

              {/* SURVEY ID */}
              <div>
                <label className="block text-[11px] text-slate-600 font-semibold mb-1">
                  Survey ID
                </label>

                <input
                  value={surveyId}
                  readOnly
                  placeholder="Auto"
                  className="w-full border border-slate-200 bg-slate-100 rounded-md px-2 py-1.5 text-[11px] font-mono font-semibold text-slate-700 placeholder:text-slate-300 cursor-not-allowed focus:outline-none"
                />

                <FieldError
                  message={errors.surveyId}
                />
              </div>

              {/* DEPTH */}
              <div>
                <label className="block text-[11px] text-slate-600 font-semibold mb-1">
                  Depth (m)
                </label>

                <input
                  value={depth}
                  onChange={(e) => {
                    setDepth(e.target.value);
                    clearFieldError('depth');
                  }}
                  placeholder="42.5"
                  inputMode="decimal"
                  className={inputClass(
                    Boolean(errors.depth)
                  )}
                />

                <FieldError
                  message={errors.depth}
                />
              </div>

              {/* LATITUDE */}
              <div>
                <label className="block text-[11px] text-slate-600 font-semibold mb-1">
                  Latitude
                </label>

                <input
                  value={latitude}
                  onChange={(e) => {
                    setLatitude(e.target.value);
                    clearFieldError('latitude');
                  }}
                  placeholder="17.3850"
                  inputMode="decimal"
                  className={inputClass(
                    Boolean(errors.latitude)
                  )}
                />

                <FieldError
                  message={errors.latitude}
                />
              </div>

              {/* LONGITUDE */}
              <div>
                <label className="block text-[11px] text-slate-600 font-semibold mb-1">
                  Longitude
                </label>

                <input
                  value={longitude}
                  onChange={(e) => {
                    setLongitude(e.target.value);
                    clearFieldError('longitude');
                  }}
                  placeholder="78.4867"
                  inputMode="decimal"
                  className={inputClass(
                    Boolean(errors.longitude)
                  )}
                />

                <FieldError
                  message={errors.longitude}
                />
              </div>
            </div>

          </div>
        </div>

        {/* ACTIONS */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 bg-slate-50/70 border-t border-slate-100">

          <div className="text-[11px] text-slate-500">
            {!hasActiveSurvey
              ? 'Select a survey to enable analysis.'
              : !selectedFile
              ? 'Upload a scan above to enable analysis.'
              : 'Complete the survey metadata and run the detection pipeline.'}
          </div>

          <div className="flex gap-2">

            <button
              onClick={() => analyzeFile()}
              disabled={!canAnalyze}
              className="btn py-1.5 px-3 text-[12px] disabled:opacity-50"
            >
              Reprocess
            </button>

            <button
              onClick={() => analyzeFile()}
              disabled={!canAnalyze}
              className="btn primary py-1.5 px-3.5 text-[12px] disabled:opacity-50 flex items-center gap-1.5"
            >
              {isAnalyzing ? (
                <>
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="w-3.5 h-3.5 animate-spin"
                  >
                    <polyline points="23 4 23 10 17 10" />
                    <polyline points="1 20 1 14 7 14" />
                  </svg>

                  Processing…
                </>
              ) : (
                <>
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="w-3.5 h-3.5"
                  >
                    <circle
                      cx="12"
                      cy="12"
                      r="9"
                    />

                    <circle
                      cx="12"
                      cy="12"
                      r="4.2"
                    />

                    <line
                      x1="12"
                      y1="12"
                      x2="18"
                      y2="6"
                    />
                  </svg>

                  Analyze
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* PIPELINE */}
      <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-sm">

        <div className="flex items-center justify-between mb-2">

          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse" />

            <span className="text-[12px] font-semibold text-slate-800">
              Preprocessing & AI Detection Pipeline
            </span>
          </div>

          <span className="text-[10px] font-mono bg-teal-50 border border-teal-200 text-teal-700 px-2 py-0.5 rounded-full">
            7 Stages · High Recall
          </span>
        </div>

        <div className="flex items-center justify-between gap-1 overflow-x-auto pb-1">

          {PREPROCESSING_STAGES.map(
            (stageName, idx) => {
              const isCompleted =
                activePipelineStage > idx ||
                (!isAnalyzing &&
                  detections.length > 0);

              const isActive =
                isAnalyzing &&
                activePipelineStage === idx;

              return (
                <React.Fragment
                  key={`pipeline-stage-${idx}-${stageName}`}
                >
                  <div className="flex-1 min-w-[70px] text-center">

                    <div
                      className={`w-7 h-7 rounded-full mx-auto mb-1 flex items-center justify-center font-mono text-[10px] font-bold border transition-all ${
                        isActive
                          ? 'bg-teal-500 border-teal-400 text-white shadow-sm ring-2 ring-teal-200 scale-105'
                          : isCompleted
                          ? 'bg-teal-500 border-teal-500 text-white'
                          : 'bg-slate-50 border-slate-200 text-slate-400'
                      }`}
                    >
                      {isCompleted ? (
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          className="w-3.5 h-3.5"
                        >
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      ) : (
                        idx + 1
                      )}
                    </div>

                    <div
                      className={`text-[9.5px] font-mono truncate ${
                        isActive ||
                        isCompleted
                          ? 'text-slate-800 font-semibold'
                          : 'text-slate-400'
                      }`}
                    >
                      {stageName}
                    </div>
                  </div>

                  {idx <
                    PREPROCESSING_STAGES.length -
                      1 && (
                    <div
                      className={`h-[2px] flex-1 max-w-[24px] ${
                        isCompleted
                          ? 'bg-teal-400'
                          : 'bg-slate-200'
                      }`}
                    />
                  )}
                </React.Fragment>
              );
            }
          )}
        </div>

        <div className="text-[11px] font-mono text-slate-500 bg-slate-50 border border-slate-100 rounded-lg px-2.5 py-1.5 mt-2 flex items-center justify-between">

          <span className="truncate">
            {statusMessage}
          </span>

          <span className="text-teal-700 font-semibold ml-2 flex-shrink-0">
            {isAnalyzing
              ? 'RUNNING'
              : detections.length > 0
              ? `${detections.length} DETECTIONS`
              : 'READY'}
          </span>
        </div>
      </div>

      {/* SONAR VISUALIZER */}
      <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-sm">

        <div className="flex items-center justify-between mb-2">
          <div className="text-[12px] font-semibold text-slate-800">
            Acoustic Sonar Visualizer
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">

          {/* RAW */}
          <div className="bg-[#030d12] border border-slate-800 rounded-lg p-2 shadow-sm">

            <div className="flex justify-between text-[9px] font-mono text-slate-400 mb-1 px-0.5">
              <span>RAW SONAR</span>

              <span className="text-teal-400 truncate max-w-[120px]">
                {rawTag}
              </span>
            </div>

            <div className="relative h-28 sm:h-32 rounded overflow-hidden bg-[#02090d] flex items-center justify-center">

              {isAnalyzing && (
                <div className="absolute inset-x-0 h-6 bg-gradient-to-b from-transparent via-teal-400/20 to-transparent animate-scan" />
              )}

              {uploadedImageSrc ? (
                <img
                  src={uploadedImageSrc}
                  alt="Raw sonar"
                  className="relative w-full h-full object-contain"
                />
              ) : (
                <span className="relative text-[9.5px] font-mono text-slate-500">
                  NO SCAN LOADED
                </span>
              )}
            </div>
          </div>

          {/* PREPROCESSED */}
          <div className="bg-[#030d12] border border-slate-800 rounded-lg p-2 shadow-sm">

            <div className="flex justify-between text-[9px] font-mono text-slate-400 mb-1 px-0.5">
              <span>PREPROCESSED</span>

              <span className="text-teal-400">
                CLAHE + BILATERAL DENOISE
              </span>
            </div>

            <div className="relative h-28 sm:h-32 rounded overflow-hidden bg-[#02090d] flex items-center justify-center">

              {isAnalyzing && (
                <div className="absolute inset-x-0 h-6 bg-gradient-to-b from-transparent via-teal-400/20 to-transparent animate-scan" />
              )}

              {processedImageSrc ? (
                <img
                  src={processedImageSrc}
                  alt="Preprocessed sonar"
                  className="relative w-full h-full object-contain"
                />
              ) : (
                <span className="relative text-[9.5px] font-mono text-slate-500">
                  {isAnalyzing
                    ? 'PROCESSING…'
                    : 'WAITING FOR PREPROCESSING'}
                </span>
              )}
            </div>
          </div>

          {/* AI DETECTION */}
          <div className="bg-[#030d12] border border-slate-800 rounded-lg p-2 shadow-sm">

            <div className="flex justify-between text-[9px] font-mono text-slate-400 mb-1 px-0.5">
              <span>AI DETECTION</span>

              <span className="text-[#00ff66] font-bold">
                {detections.length} TARGET
                {detections.length === 1
                  ? ''
                  : 'S'}
              </span>
            </div>

            <div className="relative h-28 sm:h-32 rounded overflow-hidden bg-[#02090d] flex items-center justify-center">

              {isAnalyzing && (
                <div className="absolute inset-x-0 h-6 bg-gradient-to-b from-transparent via-teal-400/20 to-transparent animate-scan" />
              )}

              {detectionImageSrc ? (
                <div className="relative w-full h-full flex items-center justify-center">

                  <img
                    src={detectionImageSrc}
                    alt="YOLO detected sonar targets"
                    className="w-full h-full object-contain"
                  />

                </div>
              ) : (
                <span className="relative text-[9.5px] font-mono text-slate-500">
                  {isAnalyzing
                    ? 'RUNNING YOLO INFERENCE…'
                    : 'WAITING FOR INFERENCE'}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* RISK + DETECTIONS */}
      <div className="space-y-3">

        {detections.length > 0 && (
          <div className="bg-white border border-slate-200 rounded-xl p-3 shadow-sm flex flex-wrap items-center justify-between gap-3">

            <div className="flex items-center gap-3">

              <div
                className={`w-12 h-12 rounded-lg flex flex-col items-center justify-center font-mono border ${overallRiskBadge.badgeClass}`}
              >
                <span className="text-[17px] font-bold leading-none">
                  {overallRisk.overallRiskScore}
                </span>

                <span className="text-[8px] font-bold tracking-widest mt-0.5">
                  SCORE
                </span>
              </div>

              <div>

                <div className="flex items-center gap-2">

                  <span className="text-[11px] font-mono text-slate-400 uppercase">
                    Overall Risk:
                  </span>

                  <span
                    className={`px-2 py-0.5 rounded-full font-mono text-[10.5px] font-bold border ${overallRiskBadge.badgeClass}`}
                  >
                    {overallRisk.overallRiskLevel}
                  </span>
                </div>

                <div className="text-[12px] text-slate-600 mt-0.5">
                  High:{' '}
                  <b className="text-rose-600">
                    {overallRisk.highCount}
                  </b>{' '}
                  · Medium:{' '}
                  <b className="text-amber-600">
                    {overallRisk.mediumCount}
                  </b>{' '}
                  · Low:{' '}
                  <b className="text-emerald-600">
                    {overallRisk.lowCount}
                  </b>{' '}
                  · Total:{' '}
                  <b>{detections.length}</b>
                </div>
              </div>
            </div>

            <button
              onClick={() =>
                onNavigate('review')
              }
              className="btn primary bg-teal-600 hover:bg-teal-700 text-white font-medium text-[11.5px] py-1.5 px-3"
            >
              Open Detection Review (
              {detections.length} pending) →
            </button>
          </div>
        )}

        {/* NO DETECTIONS */}
        {detections.length === 0 ? (
          <div className="bg-white border border-dashed border-slate-200 rounded-xl p-6 text-center text-[12px] text-slate-400">
            No objects detected yet. Run the pipeline to display YOLO target cards.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">

            {/* FIXED: SAFE UNIQUE KEYS */}
            {detections.map((detection, index) => {
              const confCat =
                getConfidenceCategory(
                  detection.confidence
                );

              const riskInfo =
                getRiskBadgeStyle(
                  detection.risk_level
                );

              return (
                <div
                  key={
                    detection.id ??
                    detection.detection_id ??
                    `detection-card-${index}`
                  }
                  className="bg-white border border-slate-200 rounded-xl p-3 shadow-sm hover:shadow-md transition-shadow duration-200 space-y-2"
                >

                  <div className="flex items-center justify-between gap-1">

                    <div className="flex items-center gap-1.5 min-w-0">

                      <span className="w-2.5 h-2.5 rounded-full bg-[#00ff44] shadow-[0_0_6px_rgba(0,255,68,0.7)] flex-shrink-0" />

                      <span className="text-[13px] font-bold text-slate-800 truncate">
                        {detection.object_class}
                      </span>

                      <span className="font-mono text-[10px] text-teal-700 border border-teal-200 bg-teal-50 rounded-full px-1.5 py-0.2 truncate">
                        {detection.full_identifier ||
                          `${activeSurveyCode || 'SURV-001'} / ${detection.detection_id}`}
                      </span>
                    </div>

                    <div className="flex items-center gap-1 font-mono text-[10px] flex-shrink-0">

                      <span
                        className={`px-1.5 py-0.2 rounded border font-semibold ${confCat.badgeClass}`}
                      >
                        {confCat.category}
                      </span>

                      <span
                        className={`px-1.5 py-0.2 rounded border font-semibold ${riskInfo.badgeClass}`}
                      >
                        RISK: {detection.risk_level}{' '}
                        ({detection.risk_score})
                      </span>

                      <span className="uppercase text-slate-400">
                        {detection.status}
                      </span>
                    </div>
                  </div>

                  {confCat.warning && (
                    <div className="bg-rose-50 border border-rose-200 text-rose-700 rounded px-2 py-1 text-[10.5px] font-mono flex items-center gap-1.5">

                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        className="w-3.5 h-3.5 text-rose-500 flex-shrink-0"
                      >
                        <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />

                        <line
                          x1="12"
                          y1="9"
                          x2="12"
                          y2="13"
                        />
                      </svg>

                      <span>
                        {confCat.warning}
                      </span>
                    </div>
                  )}

                  {/* ATTRIBUTES */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] bg-slate-50/70 p-2 rounded border border-slate-100 font-mono">

                    <div>
                      <span className="text-slate-400 block text-[9px]">
                        CONFIDENCE
                      </span>

                      <span className="font-semibold text-slate-800">
                        {formatConfidence(
                          detection.confidence
                        )}
                      </span>
                    </div>

                    <div>
                      <span className="text-slate-400 block text-[9px]">
                        DEPTH
                      </span>

                      <span className="font-semibold text-slate-800">
                        {detection.depth !=
                        null
                          ? `${detection.depth} m`
                          : '—'}
                      </span>
                    </div>

                    <div>
                      <span className="text-slate-400 block text-[9px]">
                        LATITUDE
                      </span>

                      <span className="text-slate-700">
                        {formatCoordinate(
                          detection.latitude
                        )}
                      </span>
                    </div>

                    <div>
                      <span className="text-slate-400 block text-[9px]">
                        LONGITUDE
                      </span>

                      <span className="text-slate-700">
                        {formatCoordinate(
                          detection.longitude
                        )}
                      </span>
                    </div>
                  </div>

                  {/* ACTIONS */}
                  <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-[11px]">

                    <div className="flex gap-1.5">

                      <button
                        onClick={() =>
                          handleReviewQuick(
                            detection,
                            'confirmed'
                          )
                        }
                        disabled={
                          detection.status ===
                          'confirmed'
                        }
                        className="btn confirm py-1 px-2 text-[10.5px] disabled:opacity-40"
                      >
                        ✓ Accept
                      </button>

                      <button
                        onClick={() =>
                          handleReviewQuick(
                            detection,
                            'rejected'
                          )
                        }
                        disabled={
                          detection.status ===
                          'rejected'
                        }
                        className="btn reject py-1 px-2 text-[10.5px] disabled:opacity-40"
                      >
                        ✕ Reject
                      </button>

                      <button
                        onClick={() =>
                          handleReviewQuick(
                            detection,
                            'flagged'
                          )
                        }
                        disabled={
                          detection.status ===
                          'flagged'
                        }
                        className="btn investigate py-1 px-2 text-[10.5px] disabled:opacity-40"
                      >
                        ⚑ Flag
                      </button>
                    </div>

                    <button
                      onClick={() =>
                        handleOpenReview(
                          detection
                        )
                      }
                      className="btn secondary font-medium text-[11px] py-1 px-2.5 flex items-center gap-1 hover:bg-teal-50 hover:text-teal-700 hover:border-teal-300"
                    >
                      Open review →
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
};