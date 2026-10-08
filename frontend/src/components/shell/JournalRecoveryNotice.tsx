import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { journalRecoveryStatus } from "@/lib/rehearsal/maintenance";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";

export function JournalRecoveryNotice({ locale }: { locale: UiLocale }) {
  const { pathname } = useLocation();
  const [pending, setPending] = useState(false);
  useEffect(() => {
    let active = true;
    const refresh = () => { void journalRecoveryStatus().then(status => { if (active) setPending(!!status.transaction || !!status.recovery_error); }).catch(() => {}); };
    refresh(); window.addEventListener("focus", refresh);
    return () => { active = false; window.removeEventListener("focus", refresh); };
  }, [pathname]);
  if (!pending) return null;
  const t = createTranslator(locale);
  return <aside role="status" data-testid="journal-recovery-notice"><p>{t("journal.recovery_needed")} <Link to="/settings">{t("journal.check_recovery")}</Link></p></aside>;
}
