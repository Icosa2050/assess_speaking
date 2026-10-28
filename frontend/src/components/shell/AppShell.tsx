import { Component, Suspense, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { SEMANTIC_IDS, semanticAttributes } from "@/lib/i18n";
import { isPracticePath } from "@/lib/navigation";
import { Card } from "@/components/ui/Card";
import { Icon, type IconName } from "@/components/ui/Icon";

import { JournalRecoveryNotice } from "./JournalRecoveryNotice";
import styles from "./AppShell.module.css";

type RouteItem = {
  href: string;
  label: string;
  disabled?: boolean;
  disabledReason?: string;
};

type NavGroup = {
  id: string;
  items: RouteItem[];
  label?: string;
};

type AppShellProps = {
  locale?: import("@/lib/state/sessionDraft").UiLocale;
  reloadLabel?: string;
  appName: string;
  navGroups: NavGroup[];
  navAriaLabel: string;
  skipLabel?: string;
  shellBody: string;
};

class RouteRecovery extends Component<{ children: ReactNode; label: string }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    return this.state.failed
      ? <div role="alert"><button type="button" style={{ font: "inherit" }} onClick={() => window.location.reload()}>{this.props.label}</button></div>
      : this.props.children;
  }
}

const routeSemanticAttributes = (href: string): Record<string, string> => {
  switch (href) {
    case "/history":
      return semanticAttributes(SEMANTIC_IDS.home.openHistory);
    case "/library":
      return semanticAttributes(SEMANTIC_IDS.home.openLibrary);
    case "/guide":
      return semanticAttributes(SEMANTIC_IDS.home.openGuide);
    case "/settings":
      return semanticAttributes(SEMANTIC_IDS.home.openSettings);
    default:
      return {};
  }
};

const routeIcons: Record<string, IconName> = {
  "/": "target",
  "/history": "history",
  "/library": "guide",
  "/guide": "guide",
  "/settings": "settings",
};

export const AppShell = ({
  locale = "en",
  reloadLabel = "Reload page",
  appName,
  navGroups,
  navAriaLabel,
  skipLabel = "Skip to content",
  shellBody,
}: AppShellProps) => {
  const location = useLocation();
  const primaryGroups = navGroups.filter((group) => group.id !== "preferences" && group.id !== "context");
  const contextGroups = navGroups.filter((group) => group.id === "context");
  const footerGroups = navGroups.filter((group) => group.id === "preferences");
  const renderRouteLink = (route: RouteItem) => route.disabled ? (
    <span key={route.href} className={styles.navDisabledItem}>
      <a
        className={`${styles.navLink} ${styles.navLinkDisabled}`}
        href={route.href}
        tabIndex={0}
        aria-disabled="true"
        aria-describedby={route.disabledReason ? `nav-disabled-${route.href.slice(1)}` : undefined}
        onClick={(event) => event.preventDefault()}
      >
        {route.label}
      </a>
      {route.disabledReason ? (
        <small className={styles.navDisabledReason} id={`nav-disabled-${route.href.slice(1)}`}>
          {route.disabledReason}
        </small>
      ) : null}
    </span>
  ) : (
    <NavLink
      key={route.href}
      to={route.href}
      state={route.href === "/settings"
        ? location.pathname === "/settings"
          ? location.state
          : { from: location.pathname.slice(1) || "home" }
        : route.href === "/" || route.href === "/guide"
          ? { from: location.pathname }
        : undefined}
      className={({ isActive }) => `${styles.navLink} ${isActive ? styles.navLinkActive : ""} ${route.href === "/" && location.pathname !== "/" && isPracticePath(location.pathname) ? styles.navLinkSectionActive : ""}`}
      {...routeSemanticAttributes(route.href)}
    >
      {routeIcons[route.href] ? <Icon name={routeIcons[route.href]} size={20} /> : null}
      {route.label}
    </NavLink>
  );

  return (
    <div className={styles.shell}>
      <a
        className={styles.skipLink}
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        {skipLabel}
      </a>
      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.brandBlock}>
            <h1 className={styles.title}>{appName}</h1>
            <p className={styles.body}>{shellBody}</p>
          </div>
        </header>

        <JournalRecoveryNotice locale={locale} />
        <div className={styles.layout}>
          <Card as="div" className={styles.sidebar}>
            <nav
              aria-label={navAriaLabel}
              className={styles.nav}
            >
              <div className={styles.navPrimary}>
                {primaryGroups.map((group) => (
                  <div
                    key={group.id}
                    className={styles.navGroup}
                    data-nav-group={group.id}
                  >
                    {group.label ? (
                      <span className={styles.navGroupLabel}>{group.label}</span>
                    ) : null}
                    <div className={styles.navLinks}>
                      {group.items.map(renderRouteLink)}
                    </div>
                  </div>
                ))}
              </div>
              {contextGroups.map((group) => (
                <div key={group.id} className={styles.navContext} data-nav-group={group.id} role="group" aria-labelledby={`nav-${group.id}-label`}>
                  {group.label ? <span className={styles.navGroupLabel} id={`nav-${group.id}-label`}>{group.label}</span> : null}
                  <div className={styles.navLinks}>{group.items.map(renderRouteLink)}</div>
                </div>
              ))}
              {footerGroups.length > 0 ? (
                <div className={styles.navFooter}>
                  {footerGroups.map((group) => (
                    <div key={group.id} className={styles.navGroup} data-nav-group={group.id} role="group" aria-labelledby={`nav-${group.id}-label`}>
                      {group.label ? <span className={styles.navGroupLabel} id={`nav-${group.id}-label`}>{group.label}</span> : null}
                      <div className={styles.navLinks}>{group.items.map(renderRouteLink)}</div>
                    </div>
                  ))}
                </div>
              ) : null}
            </nav>
          </Card>

          <main className={styles.main} id="main-content" tabIndex={-1}>
            <RouteRecovery key={location.pathname} label={reloadLabel}>
              <Suspense fallback={<p role="status" aria-busy="true">…</p>}>
                <Outlet />
              </Suspense>
            </RouteRecovery>
          </main>
        </div>
      </div>
    </div>
  );
};
