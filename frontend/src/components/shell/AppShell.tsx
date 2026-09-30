import { Component, Suspense, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { SEMANTIC_IDS, semanticAttributes } from "@/lib/i18n";

import styles from "./AppShell.module.css";

type RouteItem = {
  href: string;
  label: string;
};

type NavGroup = {
  id: string;
  items: RouteItem[];
  label?: string;
};

type AppShellProps = {
  reloadLabel?: string;
  appName: string;
  navGroups: NavGroup[];
  navAriaLabel: string;
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

export const AppShell = ({
  reloadLabel = "Reload page",
  appName,
  navGroups,
  navAriaLabel,
  shellBody,
}: AppShellProps) => {
  const location = useLocation();
  const primaryGroups = navGroups.filter((group) => group.id !== "settings");
  const footerGroups = navGroups.filter((group) => group.id === "settings");
  const renderRouteLink = (route: RouteItem) => (
    <NavLink
      key={route.href}
      to={route.href}
      state={route.href === "/settings"
        ? location.pathname === "/settings"
          ? location.state
          : { from: location.pathname.slice(1) || "home" }
        : undefined}
      className={({ isActive }) => `${styles.navLink} ${isActive ? styles.navLinkActive : ""}`}
      {...routeSemanticAttributes(route.href)}
    >
      {route.label}
    </NavLink>
  );

  return (
    <div className={styles.shell}>
      <div className={styles.container}>
        <header className={styles.header}>
          <div className={styles.brandBlock}>
            <h1 className={styles.title}>{appName}</h1>
            <p className={styles.body}>{shellBody}</p>
          </div>
        </header>

        <div className={styles.layout}>
          <aside className={styles.sidebar}>
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
              {footerGroups.length > 0 ? (
                <div className={styles.navFooter}>
                  {footerGroups.flatMap((group) => group.items).map(renderRouteLink)}
                </div>
              ) : null}
            </nav>
          </aside>

          <main className={styles.main}>
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
