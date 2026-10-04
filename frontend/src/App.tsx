import { lazy, useEffect, useMemo } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { SavedLocale } from "@/components/shell/SavedLocale";
import { AppShell } from "@/components/shell/AppShell";
import { apiClient } from "@/lib/api/client";
import { createTranslator, resolveUiLocale } from "@/lib/i18n";
import { queryKeys } from "@/lib/query/queryClient";
import { AppStoreProvider, useAppStore } from "@/lib/state/appStore";
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
  hideFromConfiguredNav?: boolean;
  navGroup: NavGroupId;
  path: string;
  navKey: string;
  titleKey: string;
  bodyKey: string;
};

type NavGroupId = "practice" | "progress" | "discover" | "settings";

type LocalizedRoute = {
  path: string;
  hideFromConfiguredNav?: boolean;
  navGroup: NavGroupId;
  navLabel: string;
  title: string;
  body: string;
};

const routeDefinitions: AppRouteDefinition[] = [
  { path: "/", navGroup: "practice", navKey: "nav.home", titleKey: "home.title", bodyKey: "home.body" },
  {
    path: "/runtime-setup",
    hideFromConfiguredNav: true,
    navGroup: "practice",
    navKey: "home.runtime_setup_button",
    titleKey: "runtime_setup.title",
    bodyKey: "runtime_setup.body",
  },
  { path: "/session-setup", navGroup: "practice", navKey: "nav.setup", titleKey: "setup.title", bodyKey: "setup.body" },
  { path: "/speak", navGroup: "practice", navKey: "nav.speak", titleKey: "speak.title", bodyKey: "speak.body" },
  { path: "/rehearsal", navGroup: "practice", navKey: "rehearsal.title", titleKey: "rehearsal.title", bodyKey: "rehearsal.body" },
  { path: "/review", navGroup: "practice", navKey: "nav.review", titleKey: "review.title", bodyKey: "review.body" },
  { path: "/history", navGroup: "progress", navKey: "nav.history", titleKey: "history.title", bodyKey: "history.body" },
  { path: "/library", navGroup: "discover", navKey: "nav.library", titleKey: "library.title", bodyKey: "library.body" },
  { path: "/guide", navGroup: "discover", navKey: "nav.guide", titleKey: "guide.title", bodyKey: "guide.body" },
  { path: "/settings", navGroup: "settings", navKey: "nav.settings", titleKey: "settings.title", bodyKey: "settings.body" },
];

const navGroupDefinitions: Array<{ id: NavGroupId; labelKey?: string }> = [
  { id: "practice", labelKey: "nav.group_practice" },
  { id: "progress", labelKey: "nav.group_progress" },
  { id: "discover", labelKey: "nav.group_discover" },
  { id: "settings" },
];

export const AppFrame = () => {
  const location = useLocation();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const setupComplete = useAppStore((state) => state.preferences.setupComplete);
  const translate = useMemo(() => createTranslator(locale), [locale]);
  const runtimeQuery = useQuery({
    queryKey: queryKeys.runtime,
    queryFn: () => apiClient.getRuntime(),
  });
  const runtimeConfigured = Boolean(runtimeQuery.data?.configured);
  const navSetupComplete = setupComplete || runtimeConfigured;

  const localizedRoutes = useMemo<LocalizedRoute[]>(
    () =>
      routeDefinitions.map((route) => ({
        path: route.path,
        hideFromConfiguredNav: route.hideFromConfiguredNav,
        navGroup: route.navGroup,
        navLabel: translate(route.navKey),
        title: translate(route.titleKey),
        body: translate(route.bodyKey),
      })),
    [translate],
  );

  const navGroups = useMemo(
    () =>
      navGroupDefinitions
        .map((group) => ({
          id: group.id,
          label: group.labelKey ? translate(group.labelKey) : undefined,
          items: localizedRoutes
            .filter(
              (route) =>
                route.navGroup === group.id &&
                !(route.hideFromConfiguredNav && navSetupComplete),
            )
            .map((route) => ({
              href: route.path,
              label: route.navLabel,
            })),
        }))
        .filter((group) => group.items.length > 0),
    [localizedRoutes, navSetupComplete, translate],
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
          <AppShell
            reloadLabel={translate("nav.reload_page")}
            appName={translate("home.title")}
            navAriaLabel={translate("nav.main_navigation")}
            navGroups={navGroups}
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
      <AppFrame />
    </AppStoreProvider>
  );
}
