// App.jsx – main application component with workflow integration
import React, { useEffect, useState } from 'react';
import { useToast } from './context/ToastContext';
import { ToastContainer } from './components/common/ToastContainer';
import { Sidebar } from './components/layout/Sidebar';
import { Topbar } from './components/layout/Topbar';

// Page Views
import { Landing } from './components/pages/Landing';
import { DashboardPage } from './components/pages/DashboardPage';
import { SurveysPage } from './components/pages/SurveysPage';
import { SonarPage } from './components/pages/SonarPage';
import { ReviewPage } from './components/pages/ReviewPage';
import { MapPage } from './components/pages/MapPage';
import { CleanupPage } from './components/pages/CleanupPage';
import { VerifyPage } from './components/pages/VerifyPage';
import { ReportsPage } from './components/pages/ReportsPage';
import { SettingsPage } from './components/pages/SettingsPage';

import api, { buildDashboardOverview } from './services/api';

export const App = () => {
  // ToastProvider must wrap App from main.jsx
  const { showToast } = useToast();
  const [showLanding, setShowLanding] = useState(true);
  const [activePage, setActivePage] = useState('dashboard');
  const [mobileOpen, setMobileOpen] = useState(false);

  const [surveys, setSurveys] = useState([]);
  const [rawDetections, setRawDetections] = useState([]);

  const [overviewData, setOverviewData] = useState({
    loading: true,
    error: null,
    activeSurvey: null,
    stats: {},
    riskDistribution: null,
    recentDetections: [],
    objectDistribution: [],
  });

  const [currentSurveyId, setCurrentSurveyId] = useState(null);

  // Most recent analysis result used by Detection Review
  const [latestAnalysis, setLatestAnalysis] = useState(null);
  const [selectedReviewDetectionId, setSelectedReviewDetectionId] = useState(null);

  // --------------------------------------------------
  // LOAD INITIAL DASHBOARD DATA
  // --------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    const loadDashboard = async () => {
      try {
        const [detections, surveyData] = await Promise.all([
          api.getDetections(),
          api.getSurveys(),
        ]);

        if (cancelled) return;

        const det = Array.isArray(detections) ? detections : [];
        const surveysList = Array.isArray(surveyData) ? surveyData : [];

        // Determine latest active survey ID from backend
        let initialSurveyId = null;
        if (surveysList.length > 0) {
          const sorted = [...surveysList].sort((a, b) => (Number(a.id) || 0) - (Number(b.id) || 0));
          initialSurveyId = sorted[0].id;
        }

        const nextOverview = buildDashboardOverview(
          det,
          surveysList,
          initialSurveyId
        );

        setSurveys(surveysList);
        setRawDetections(det);
        if (initialSurveyId) {
          setCurrentSurveyId(initialSurveyId);
        }

        setOverviewData({
          ...nextOverview,
          loading: false,
          error: null,
        });
      } catch (err) {
        if (cancelled) return;

        setOverviewData((prev) => ({
          ...prev,
          loading: false,
          error:
            err?.message ||
            'Unable to connect to backend.',
        }));
      }
    };

    loadDashboard();

    return () => {
      cancelled = true;
    };
  }, []);

  // Synchronize dashboard overview with live survey/detection telemetry whenever raw data changes
  useEffect(() => {
    if (overviewData.loading) return;
    const nextOverview = buildDashboardOverview(rawDetections, surveys, currentSurveyId);
    setOverviewData((prev) => ({
      ...nextOverview,
      loading: false,
      error: prev.error,
    }));
  }, [rawDetections, surveys, currentSurveyId]);

  // --------------------------------------------------
  // NAVIGATION
  // --------------------------------------------------

  const handleNavigate = (pageId, params = {}) => {
    setActivePage(pageId);
    if (params && params.selectedDetectionId !== undefined) {
      setSelectedReviewDetectionId(params.selectedDetectionId);
    } else if (pageId === 'review') {
      setSelectedReviewDetectionId(null);
    }
    setMobileOpen(false);

    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };
  const handleEnterApp = () => {
  setShowLanding(false);
  setActivePage('dashboard');
  window.scrollTo({ top: 0 });
};

  // --------------------------------------------------
  // ANALYSIS COMPLETE
  // --------------------------------------------------

  const handleAnalysisComplete = (result, normalized = []) => {
    // Persist survey ID
    if (result?.survey_id) {
      setCurrentSurveyId(result.survey_id);
    }

    // Store complete analysis for Review page with exact normalized data
    const finalAnalysis = {
      ...result,
      detections: normalized,
      detectionImage: result.detection_image || result.detectionImage,
      processedImage: result.processed_image || result.processedImage,
      surveyId: result.survey_id,
      survey: result.survey,
    };
    setLatestAnalysis(finalAnalysis);

    // Add only actual normalized detections to raw detections
    if (Array.isArray(normalized) && normalized.length > 0) {
      setRawDetections((prev) => [
        ...normalized,
        ...prev.filter((d) => !normalized.some((n) => String(n.id) === String(d.id))),
      ]);
    }
  };

  // --------------------------------------------------
  // DETECTION REVIEW
  // --------------------------------------------------

  const handleDetectionReviewed = (
    detectionId,
    newStatus
  ) => {
    setRawDetections((prev) =>
      prev.map((det) => {
        const id =
          det.id ??
          det.detection_id;

        return id === detectionId
          ? {
              ...det,
              status: newStatus,
            }
          : det;
      })
    );

    // Keep latest analysis synchronized if possible
    setLatestAnalysis((prev) => {
      if (!prev) return prev;

      return {
        ...prev,
        detections: Array.isArray(prev.detections)
          ? prev.detections.map((det) => {
              const id =
                det.id ??
                det.detection_id;

              return id === detectionId
                ? {
                    ...det,
                    status: newStatus,
                  }
                : det;
            })
          : prev.detections,
      };
    });
  };

  // --------------------------------------------------
  // CREATE NEW SURVEY
  // --------------------------------------------------

  const handleCreateSurvey = async (payload) => {
    try {
      const created = await api.createSurvey(payload);

      if (!created?.id) {
        throw new Error(
          'Survey was created but no survey ID was returned by the backend.'
        );
      }

      // Add newly created survey
      setSurveys((prev) => [
        created,
        ...prev,
      ]);

      // IMPORTANT:
      // Keep this survey ID for Sonar → Review → Map → Cleanup
      setCurrentSurveyId(created.id);

      // Immediately continue to Sonar Analysis
      handleNavigate('sonar');

      showToast({
        type: 'success',
        message: 'Survey created successfully. Ready for sonar analysis.',
      });

      return created;
    } catch (err) {
      showToast({
        type: 'error',
        message:
          err?.message ||
          'Failed to create survey.',
      });

      throw err;
    }
  };

  // --------------------------------------------------
  // 17. CONFIRMED DETECTIONS ONLY (For Risk Map & Cleanup Priority)
  // --------------------------------------------------

  const confirmedDetections = rawDetections.filter((d) => {
    const st = String(d.status || d.cleanup_status || '').toLowerCase();
    return st === 'confirmed' || st === 'verified' || st === 'accepted' || st === 'cleaned';
  });

  const mapDetections = confirmedDetections;

  // Active survey memo
  const activeSurvey = React.useMemo(() => {
    if (!currentSurveyId) {
      if (surveys.length > 0) {
        const sorted = [...surveys].sort((a, b) => (Number(a.id) || 0) - (Number(b.id) || 0));
        return sorted[0];
      }
      return null;
    }
    return surveys.find((s) => String(s.id) === String(currentSurveyId)) || null;
  }, [surveys, currentSurveyId]);

  const activeSurveyCode = activeSurvey
    ? `SURV-${String(activeSurvey.id).padStart(3, '0')}`
    : 'NO SURVEY';
  if (showLanding) {
  return (
    <>
      <Landing onEnter={handleEnterApp} />
      <ToastContainer />
    </>
  );
}
  // --------------------------------------------------
  // RENDER
  // --------------------------------------------------

  return (
    <div className="app-layout flex min-h-screen bg-white text-gray-900 font-ui selection:bg-teal-300 selection:text-gray-900 overflow-x-hidden">

      {/* Sidebar Navigation */}
      <Sidebar
        activePage={activePage}
        onNavigate={handleNavigate}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      {/* Main Application Area */}
      <div className="app-main-area flex-1 min-w-0 flex flex-col">

        <Topbar
          activePage={activePage}
          onToggleMobile={() =>
            setMobileOpen(!mobileOpen)
          }
          surveyCode={activeSurveyCode}
        />

        <main className="flex-1 p-4 sm:p-6 md:p-8 max-w-[1360px] w-full mx-auto pb-16">

          {/* Dashboard */}
          {activePage === 'dashboard' && (
            <DashboardPage
              onNavigate={handleNavigate}
              overviewData={overviewData}
              surveys={surveys}
              currentSurveyId={currentSurveyId}
              onSelectSurvey={setCurrentSurveyId}
            />
          )}

          {/* Surveys */}
          {activePage === 'surveys' && (
            <SurveysPage
              onNavigate={handleNavigate}
              surveys={surveys}
              detections={rawDetections}
              onCreateSurvey={handleCreateSurvey}
              currentSurveyId={currentSurveyId}
              onSelectSurvey={setCurrentSurveyId}
            />
          )}

          {/* Sonar Analysis */}
          {activePage === 'sonar' && (
            <SonarPage
              onNavigate={handleNavigate}
              onAnalysisComplete={handleAnalysisComplete}
              onDetectionReviewed={handleDetectionReviewed}
              initialSurveyId={currentSurveyId}
              activeSurvey={activeSurvey}
              surveys={surveys}
              onSelectSurvey={setCurrentSurveyId}
            />
          )}

          {/* Detection Review / Human Verification */}
          {activePage === 'review' && (
            <ReviewPage
              onNavigate={handleNavigate}
              analysis={latestAnalysis}
              allDetectionsList={rawDetections}
              initialSelectedDetectionId={selectedReviewDetectionId}
              onClearSelection={() => setSelectedReviewDetectionId(null)}
              currentSurveyId={currentSurveyId}
              activeSurvey={activeSurvey}
              onReviewed={handleDetectionReviewed}
            />
          )}

          {/* Risk Map (CONFIRMED detections only - Req 17) */}
          {activePage === 'map' && (
            <MapPage
              detections={confirmedDetections}
              surveys={surveys}
              activeSurvey={activeSurvey}
              onNavigate={handleNavigate}
            />
          )}

          {/* Cleanup Priority (CONFIRMED detections only - Req 17) */}
          {activePage === 'cleanup' && (
            <CleanupPage
              detections={confirmedDetections}
              surveys={surveys}
              activeSurvey={activeSurvey}
              onNavigate={handleNavigate}
            />
          )}

          {/* Verification */}
          {activePage === 'verify' && (
            <VerifyPage
              confirmedDetections={confirmedDetections}
              onNavigate={handleNavigate}
            />
          )}

          {/* Reports */}
          {activePage === 'reports' && (
            <ReportsPage
              detections={rawDetections}
              surveys={surveys}
              currentSurveyId={currentSurveyId}
              onNavigate={handleNavigate}
              onSelectSurvey={setCurrentSurveyId}
            />
          )}

          {/* System Status */}
          {activePage === 'settings' && (
            <SettingsPage />
          )}

        </main>
      </div>

      {/* Global Toasts */}
      <ToastContainer />

    </div>
  );
};

export default App;