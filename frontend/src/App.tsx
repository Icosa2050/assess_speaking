import { lazy, useEffect, useMemo } from "react";
import { applyAppearance, systemPrefersDark } from "@/lib/state/appearance";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { SavedLocale } from "@/components/shell/SavedLocale";
import { AppShell } from "@/components/shell/AppShell";
import { apiClient } from "@/lib/api/client";
import { createTranslator, resolveUiLocale } from "@/lib/i18n";
import { isPracticePath } from "@/lib/navigation";
import { queryKeys } from "@/lib/query/queryClient";
import { AppStoreProvider, useAppStore } from "@/lib/state/appStore";
import { DraftRecoveryProvider } from "@/lib/state/DraftRecoveryProvider";
import { hasSetupDraft } from "@/lib/state/sessionDraft";
import { HomeRoute } from "@/routes/HomeRoute";

const RehearsalRoute = lazy(() => import("@/routes/RehearsalRoute").then(module => ({ default: module.RehearsalRoute })));
const GuideRoute = lazy(() => import("@/routes/GuideRoute").then(module => ({ default: module.GuideRoute })));
const HistoryRoute = lazy(() => import("@/routes/HistoryRoute").then(module => ({ default: module.HistoryRoute })));
const LibraryRoute = lazy(() => import("@/routes/LibraryRoute").then(module => ({ default: module.LibraryRoute })));
const ReviewRoute = lazy(() => import("@/routes/ReviewRoute").then(module => ({ default: module.ReviewRoute })));
const SessionSetupRoute = lazy(() => import("@/routes/SessionSetupRoute").then(module => ({ default: module.SessionSetupRoute })));
const SettingsRoute = lazy(() => import("@/routes/SettingsRoute").then(module => ({ default: module.SettingsRoute })));
const SpeakRoute = lazy(() => import("@/routes/SpeakRoute").then(module => ({ default: module.SpeakRoute })));
const SetupRoute = lazy(() => import("@/routes/SetupRoute").then(module => ({ default: module.SetupRoute })));

type AppRouteDefinition = {
  path: string;
  navKey: string;
  titleKey: string;
  bodyKey: string;
};

type LocalizedRoute = {
  path: string;
  navLabel: string;
  title: string;
  body: string;
};

const routeDefinitions: AppRouteDefinition[] = [
  { path: "/", navKey: "nav.home", titleKey: "home.title", bodyKey: "home.body" },
  {
    path: "/runtime-setup",
    navKey: "home.runtime_setup_button",
    titleKey: "runtime_setup.title",
    bodyKey: "runtime_setup.body",
  },
  { path: "/session-setup", navKey: "nav.setup", titleKey: "setup.title", bodyKey: "setup.body" },
  { path: "/speak", navKey: "nav.speak", titleKey: "speak.title", bodyKey: "speak.body" },
  { path: "/rehearsal", navKey: "rehearsal.title", titleKey: "rehearsal.title", bodyKey: "rehearsal.body" },
  { path: "/review", navKey: "nav.review", titleKey: "review.title", bodyKey: "review.body" },
  { path: "/history", navKey: "nav.history", titleKey: "history.title", bodyKey: "history.body" },
  { path: "/library", navKey: "nav.library", titleKey: "library.title", bodyKey: "library.body" },
  { path: "/guide", navKey: "nav.guide", titleKey: "guide.title", bodyKey: "guide.body" },
  { path: "/settings", navKey: "nav.settings", titleKey: "settings.title", bodyKey: "settings.body" },
];

export const AppFrame = () => {
  const location = useLocation();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const appearance = useAppStore((state) => state.preferences.appearance);
  useEffect(() => {
    applyAppearance(appearance, systemPrefersDark());
    if (appearance !== "system" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => applyAppearance("system", media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [appearance]);
  const hasReview = useAppStore((state) => Boolean(state.review.reportId));
  const hasDraft = useAppStore((state) => hasSetupDraft(state.draft));
  const fromRoute = typeof location.state?.from === "string" && isPracticePath(location.state.from)
    ? location.state.from
    : null;
  const translate = useMemo(() => createTranslator(locale), [locale]);
  // Keep readiness warm for guarded Speak links after a History reload and retry.
  useQuery({ queryKey: queryKeys.runtime, queryFn: () => apiClient.getRuntime() });

  const localizedRoutes = useMemo<LocalizedRoute[]>(
    () =>
      routeDefinitions.map((route) => ({
        path: route.path,
        navLabel: translate(route.navKey),
        title: translate(route.titleKey),
        body: translate(route.bodyKey),
      })),
    [translate],
  );

  const navGroups = useMemo(
    () => {
      const routeAt = (path: string) => localizedRoutes.find((route) => route.path === path)!;
      const item = (path: string, label = routeAt(path).navLabel, disabled = false) => ({ href: path, label, disabled });
      const contextualPaths = (location.pathname === "/" || location.pathname === "/guide") && fromRoute
        ? [fromRoute]
        : isPracticePath(location.pathname)
        ? location.pathname === "/runtime-setup"
          ? ["/runtime-setup", ...(hasDraft ? ["/speak"] : []), "/rehearsal"]
          : location.pathname === "/session-setup"
            ? ["/session-setup"]
            : location.pathname === "/review"
              ? ["/session-setup", "/review"]
              : ["/session-setup", location.pathname, "/review"]
        : [];
      return [
        {
          id: "primary",
          items: [
            item("/", translate("nav.group_practice")),
            item("/history", translate("nav.group_progress")),
            item("/library"),
          ],
        },
        ...(contextualPaths.length ? [{
          id: "context",
          label: fromRoute && (location.pathname === "/" || location.pathname === "/guide")
            ? translate("nav.return_to_step")
            : translate("nav.current_step"),
          items: contextualPaths.map((path) => {
            const disabled = path === "/review" && path !== location.pathname && !hasReview;
            return { ...item(path, routeAt(path).navLabel, disabled),
              disabledReason: disabled ? translate("review.guard_missing_review") : undefined };
          }),
        }] : []),
        {
          id: "preferences",
          label: translate("nav.preferences"),
          items: [item("/guide", translate("nav.how_scores_work")), item("/settings#your-data", translate("your_data.title")), item("/settings")],
        },
      ];
    },
    [fromRoute, hasDraft, hasReview, localizedRoutes, location.pathname, translate],
  );

  const currentRoute = localizedRoutes.find((route) => route.path === location.pathname) ?? localizedRoutes[0];

  useEffect(() => {
    document.documentElement.lang = resolveUiLocale(locale);
    document.title = currentRoute.title;
  }, [currentRoute.title, locale]);

  return (
    <Routes>
      <Route
        element={
          <AppShell locale={locale}
            reloadLabel={translate("nav.reload_page")}
            appName={translate("home.title")}
            navAriaLabel={translate("nav.main_navigation")}
            navGroups={navGroups}
            skipLabel={translate("nav.skip_to_content")}
            shellBody={translate("home.body")}
          />
        }
      >
        <Route index element={<HomeRoute />} />
        <Route path="runtime-setup" element={<SetupRoute />} />
        <Route path="session-setup" element={<SessionSetupRoute />} />
        <Route path="speak" element={<SpeakRoute />} />
        <Route path="rehearsal" element={<RehearsalRoute />} />
        <Route path="review" element={<ReviewRoute />} />
        <Route path="history" element={<HistoryRoute />} />
        <Route path="library" element={<LibraryRoute />} />
        <Route path="guide" element={<GuideRoute />} />
        <Route path="settings" element={<SettingsRoute />} />
      </Route>
      <Route
        path="*"
        element={<Navigate replace to="/" />}
      />
    </Routes>
  );
};

export default function App() {
  return (
    <AppStoreProvider
      store={undefined}
    >
      <SavedLocale />
      <DraftRecoveryProvider><AppFrame /></DraftRecoveryProvider>
    </AppStoreProvider>
  );
}
