import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/lib/api/client";
import type { SharingRoute, SharingSelection } from "@/lib/api/types";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";

export function useSharingRoute(selection?: SharingSelection, enabled = true) {
  return useQuery({ queryKey: ["runtime", "sharing", selection ?? null], queryFn: () => apiClient.getSharingRoute(selection), enabled, retry: false });
}
export function SharingSummary({ route, locale }: { route?: SharingRoute; locale: UiLocale }) {
  const t = createTranslator(locale);
  if (!route?.available) return <p role="status" data-testid="sharing-summary">{t("sharing.unavailable")}</p>;
  const lines = [t(route.audio.local ? "sharing.audio_local" : "sharing.audio_remote", { provider: route.audio.provider, host: route.audio.host, model: route.audio.model }),
    t(route.analysis.local ? "sharing.text_local" : "sharing.text_remote", { provider: route.analysis.provider, host: route.analysis.host, model: route.analysis.model })];
  if (route.fallback) lines.push(t("sharing.fallback", { provider: route.fallback.provider, host: route.fallback.host, model: route.fallback.model, mode: route.fallback.mode }));
  if ([route.analysis, route.fallback].some(item => item?.provider === "openrouter")) lines.push(t("sharing.openrouter"));
  return <section aria-label={t("sharing.title")} data-testid="sharing-summary"><strong>{t("sharing.title")}</strong>{lines.map(line => <p key={line}>{line}</p>)}</section>;
}
export function CurrentSharingSummary({ locale }: { locale: UiLocale }) {
  const query = useSharingRoute();
  return <SharingSummary route={query.data} locale={locale} />;
}
