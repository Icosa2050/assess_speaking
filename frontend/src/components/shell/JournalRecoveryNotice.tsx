import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { journalRecoveryStatus } from "@/lib/rehearsal/maintenance";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";

export function JournalRecoveryNotice({ locale }: { locale: UiLocale }) {
  const { pathname } = useLocation();
  const [damaged, setDamaged] = useState(false);
  const [pending, setPending] = useState(false);
  useEffect(() => {
    let active = true;
    const refresh = () => { void journalRecoveryStatus().then(status => { if (active) { setPending(!!status.transaction || !!status.recovery_error); setDamaged(!!status.recovery_error); } }).catch(() => {}); };
    refresh(); window.addEventListener("focus", refresh); window.addEventListener("journal-recovery-changed", refresh);
    return () => { active = false; window.removeEventListener("focus", refresh); window.removeEventListener("journal-recovery-changed", refresh); };
  }, [pathname]);
  if (!pending) return null;
  const t = createTranslator(locale);
  if (damaged) return <aside role="status" data-testid="journal-recovery-notice">
    <p>{t("journal.damaged_guidance")}</p>
    <ul>
      <li><Link to="/settings#journal-restore">{t("journal.restore_backup")}</Link></li>
      <li><Link to="/settings#journal-diagnostics">{t("journal.diagnostics")}</Link></li>
      <li><Link to="/settings#journal-fresh">{t("journal.fresh")}</Link></li>
    </ul>
  </aside>;
  return <aside role="status" data-testid="journal-recovery-notice"><p>{t("journal.recovery_needed")} <Link to="/settings">{t("journal.check_recovery")}</Link></p></aside>;
}
