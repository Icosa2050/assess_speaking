import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import { useAppStore } from "@/lib/state/appStore";

/** Restore the saved interface language even when opening History directly. */
export function SavedLocale() {
  const restored = useRef(false);
  const setUiLocale = useAppStore(state => state.setUiLocale);
  const settings = useQuery({
    queryKey: ["runtime", "settings"],
    queryFn: () => apiClient.getRuntimeSettings(),
  });
  useEffect(() => {
    if (restored.current || !settings.data?.ui_locale) return;
    restored.current = true;
    setUiLocale(settings.data.ui_locale);
  }, [settings.data, setUiLocale]);
  return null;
}
