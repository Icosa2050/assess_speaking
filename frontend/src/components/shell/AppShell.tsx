import { NavLink, Outlet } from "react-router-dom";

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
  onLocaleChange,
  routeItems,
  shellBody,
  shellSecondaryBody,
  shellSecondaryTitle,
}: AppShellProps) => (
  <div
    style={{
      minHeight: "100vh",
      background:
        "linear-gradient(180deg, #eef4f2 0%, #f8fbfa 48%, #f3f6f5 100%)",
      color: "#10201c",
    }}
  >
    <div
      style={{
        maxWidth: "1320px",
        margin: "0 auto",
        padding: "24px",
      }}
    >
      <header
        style={{
          display: "grid",
          gap: "1rem",
          gridTemplateColumns: "minmax(0, 1fr) auto",
          alignItems: "start",
          marginBottom: "24px",
        }}
      >
        <div style={{ display: "grid", gap: "0.5rem" }}>
          <h1
            style={{
              margin: 0,
              fontSize: "clamp(2.2rem, 3vw, 3.2rem)",
              lineHeight: 1.05,
            }}
          >
            {appName}
          </h1>
          <p
            style={{
              margin: 0,
              maxWidth: "56ch",
              lineHeight: 1.6,
              color: "#33514b",
            }}
          >
            {shellBody}
          </p>
        </div>
        <div
          style={{
            display: "grid",
            gap: "0.625rem",
            justifyItems: "end",
          }}
        >
          <span
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "#33514b",
            }}
          >
            {localeLabel}
          </span>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "0.5rem",
              justifyContent: "flex-end",
            }}
          >
            {localeOptions.map((option) => (
              <button
                key={option.code}
                type="button"
                onClick={() => onLocaleChange(option.code)}
                aria-pressed={activeLocale === option.code}
                style={{
                  padding: "0.55rem 0.85rem",
                  borderRadius: "8px",
                  border:
                    activeLocale === option.code
                      ? "1px solid rgba(15, 118, 110, 0.3)"
                      : "1px solid rgba(18, 61, 55, 0.12)",
                  backgroundColor:
                    activeLocale === option.code ? "#d7ebe5" : "rgba(255, 255, 255, 0.82)",
                  color: "#10201c",
                  cursor: "pointer",
                  font: "inherit",
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div
        style={{
          display: "grid",
          gap: "24px",
          gridTemplateColumns: "minmax(220px, 280px) minmax(0, 1fr)",
          alignItems: "start",
        }}
      >
        <aside
          style={{
            display: "grid",
            gap: "0.875rem",
            padding: "1rem",
            border: "1px solid rgba(18, 61, 55, 0.12)",
            borderRadius: "8px",
            backgroundColor: "rgba(255, 255, 255, 0.72)",
            boxShadow: "0 14px 30px rgba(16, 32, 28, 0.06)",
          }}
        >
          <p
            style={{
              margin: 0,
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "#33514b",
            }}
          >
            {shellSecondaryTitle}
          </p>
          <p
            style={{
              margin: 0,
              lineHeight: 1.55,
              color: "#4a6660",
            }}
          >
            {shellSecondaryBody}
          </p>
          <nav
            aria-label={appName}
            style={{
              display: "grid",
              gap: "0.5rem",
              marginTop: "0.5rem",
            }}
          >
            {routeItems.map((route) => (
              <NavLink
                key={route.href}
                to={route.href}
                style={({ isActive }) => ({
                  display: "block",
                  padding: "0.75rem 0.85rem",
                  borderRadius: "8px",
                  textDecoration: "none",
                  fontWeight: 600,
                  backgroundColor: isActive ? "#d7ebe5" : "rgba(255, 255, 255, 0.9)",
                  color: "#10201c",
                  border: isActive
                    ? "1px solid rgba(15, 118, 110, 0.25)"
                    : "1px solid rgba(18, 61, 55, 0.08)",
                })}
              >
                {route.label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main
          style={{
            display: "grid",
            gap: "1rem",
          }}
        >
          <Outlet />
        </main>
      </div>
    </div>
  </div>
);
