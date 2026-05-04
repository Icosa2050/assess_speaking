import { useEffect, useMemo } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import { AppShell } from "@/components/shell/AppShell";
import { createTranslator, detectPreferredUiLocale, resolveUiLocale } from "@/lib/i18n";
import { AppStoreProvider, useAppStore } from "@/lib/state/appStore";
import { SUPPORTED_UI_LOCALES, type UiLocale } from "@/lib/state/sessionDraft";
import { GuideRoute } from "@/routes/GuideRoute";
import { HomeRoute } from "@/routes/HomeRoute";
import { HistoryRoute } from "@/routes/HistoryRoute";
import { LibraryRoute } from "@/routes/LibraryRoute";
import { ReviewRoute } from "@/routes/ReviewRoute";
import { SessionSetupRoute } from "@/routes/SessionSetupRoute";
import { SettingsRoute } from "@/routes/SettingsRoute";
import { SpeakRoute } from "@/routes/SpeakRoute";
import { SetupRoute } from "@/routes/SetupRoute";

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
  { path: "/runtime-setup", navKey: "home.runtime_setup_button", titleKey: "runtime_setup.title", bodyKey: "runtime_setup.body" },
  { path: "/session-setup", navKey: "nav.setup", titleKey: "setup.title", bodyKey: "setup.body" },
  { path: "/speak", navKey: "nav.speak", titleKey: "speak.title", bodyKey: "speak.body" },
  { path: "/review", navKey: "nav.review", titleKey: "review.title", bodyKey: "review.body" },
  { path: "/history", navKey: "nav.history", titleKey: "history.title", bodyKey: "history.body" },
  { path: "/library", navKey: "nav.library", titleKey: "library.title", bodyKey: "library.body" },
  { path: "/guide", navKey: "nav.guide", titleKey: "guide.title", bodyKey: "guide.body" },
  { path: "/settings", navKey: "nav.settings", titleKey: "settings.title", bodyKey: "settings.body" },
];

export const AppFrame = () => {
  const location = useLocation();
  const locale = useAppStore((state) => state.preferences.uiLocale);
  const setLocale = useAppStore((state) => state.setUiLocale);
  const translate = createTranslator(locale);

  const localizedRoutes = useMemo<LocalizedRoute[]>(
    () =>
      routeDefinitions.map((route) => ({
        path: route.path,
        navLabel: translate(route.navKey),
        title: translate(route.titleKey),
        body: translate(route.bodyKey),
      })),
    [locale],
  );

  const currentRoute = localizedRoutes.find((route) => route.path === location.pathname) ?? localizedRoutes[0];

  const localeDisplayNames = useMemo(() => {
    if (typeof Intl.DisplayNames !== "function") {
      return Object.fromEntries(SUPPORTED_UI_LOCALES.map((code) => [code, code.toUpperCase()])) as Record<UiLocale, string>;
    }

    const displayNames = new Intl.DisplayNames([locale], { type: "language" });
    return Object.fromEntries(
      SUPPORTED_UI_LOCALES.map((code) => [code, displayNames.of(code) ?? code.toUpperCase()]),
    ) as Record<UiLocale, string>;
  }, [locale]);

  useEffect(() => {
    document.documentElement.lang = resolveUiLocale(locale);
    document.title = currentRoute.title;
  }, [currentRoute.title, locale]);

  return (
    <Routes>
      <Route
        element={
          <AppShell
            appName={translate("home.title")}
            localeLabel={translate("settings.ui_locale")}
            localeOptions={SUPPORTED_UI_LOCALES.map((code) => ({
              code,
              label: localeDisplayNames[code],
            }))}
            activeLocale={locale}
            onLocaleChange={(value) => {
              if (SUPPORTED_UI_LOCALES.includes(value as UiLocale)) {
                setLocale(value);
              }
            }}
            routeItems={localizedRoutes.map((route) => ({
              href: route.path,
              label: route.navLabel,
            }))}
            shellBody={translate("home.body")}
            shellSecondaryTitle={translate("home.secondary_title")}
            shellSecondaryBody={translate("home.secondary_body")}
          />
        }
      >
        <Route index element={<HomeRoute />} />
        <Route path="runtime-setup" element={<SetupRoute />} />
        <Route path="session-setup" element={<SessionSetupRoute />} />
        <Route path="speak" element={<SpeakRoute />} />
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
      <AppFrame />
    </AppStoreProvider>
  );
}
