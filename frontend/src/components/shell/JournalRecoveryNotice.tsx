import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { isOpenJournalPreview, journalRecoveryStatus } from "@/lib/rehearsal/maintenance";
import { createTranslator } from "@/lib/i18n";
import type { UiLocale } from "@/lib/state/sessionDraft";
import { StatusBanner } from "@/components/ui/StatusBanner";

export function JournalRecoveryNotice({ locale }: { locale: UiLocale }) {
  const { pathname } = useLocation();
  const [damaged, setDamaged] = useState(false);
  const [pending, setPending] = useState(false);
  useEffect(() => {
    let active = true;
    const refresh = () => { void journalRecoveryStatus().then(status => { if (active) { setPending(!(status.transaction && isOpenJournalPreview(status.transaction.id)) && (!!status.transaction || !!status.recovery_error)); setDamaged(!!status.recovery_error); } }).catch(() => {}); };
    refresh(); window.addEventListener("focus", refresh); window.addEventListener("journal-recovery-changed", refresh);
    return () => { active = false; window.removeEventListener("focus", refresh); window.removeEventListener("journal-recovery-changed", refresh); };
  }, [pathname]);
  if (!pending) return null;
  const t = createTranslator(locale);
  if (damaged) return <div data-testid="journal-recovery-notice">
    <StatusBanner title={t("journal.damaged_guidance")} tone="warning">
    <ul>
      <li><Link to="/settings#journal-restore">{t("journal.restore_backup")}</Link></li>
      <li><Link to="/settings#journal-diagnostics">{t("journal.diagnostics")}</Link></li>
      <li><Link to="/settings#journal-fresh">{t("journal.fresh")}</Link></li>
    </ul>
    </StatusBanner>
  </div>;
  return <div data-testid="journal-recovery-notice"><StatusBanner title={t("journal.recovery_needed")} action={<Link to="/settings">{t("journal.check_recovery")}</Link>} /></div>;
}
