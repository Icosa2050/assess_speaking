import { NavLink, Outlet } from "react-router-dom";

import styles from "./AppShell.module.css";

type LocaleOption = {
  code: string;
  label: string;
};

type RouteItem = {
  href: string;
  label: string;
};

type AppShellProps = {
  activeLocale: string;
  appName: string;
  localeLabel: string;
  localeOptions: LocaleOption[];
  navAriaLabel: string;
  onLocaleChange: (locale: string) => void;
  routeItems: RouteItem[];
  shellBody: string;
  shellSecondaryBody: string;
  shellSecondaryTitle: string;
};

export const AppShell = ({
  activeLocale,
  appName,
  localeLabel,
  localeOptions,
  navAriaLabel,
  onLocaleChange,
  routeItems,
  shellBody,
  shellSecondaryBody,
  shellSecondaryTitle,
}: AppShellProps) => (
  <div className={styles.shell}>
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.brandBlock}>
          <h1 className={styles.title}>{appName}</h1>
          <p className={styles.body}>{shellBody}</p>
        </div>
        <div className={styles.localePanel}>
          <span className={styles.localeLabel}>{localeLabel}</span>
          <div className={styles.localeOptions}>
            {localeOptions.map((option) => (
              <button
                key={option.code}
                type="button"
                onClick={() => onLocaleChange(option.code)}
                aria-pressed={activeLocale === option.code}
                className={`${styles.localeButton} ${activeLocale === option.code ? styles.localeButtonActive : ""}`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className={styles.layout}>
        <aside className={styles.sidebar}>
          <p className={styles.sidebarTitle}>{shellSecondaryTitle}</p>
          <p className={styles.sidebarBody}>{shellSecondaryBody}</p>
          <nav
            aria-label={navAriaLabel}
            className={styles.nav}
          >
            {routeItems.map((route) => (
              <NavLink
                key={route.href}
                to={route.href}
                className={({ isActive }) => `${styles.navLink} ${isActive ? styles.navLinkActive : ""}`}
              >
                {route.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main className={styles.main}>
          <Outlet />
        </main>
      </div>
    </div>
  </div>
);
