from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from corpora.lips_dataset import (
    LipsBuildConfig,
    LipsValidationConfig,
    build_lips_manifest,
    read_jsonl,
    validate_lips_manifest,
    write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "corpora" / "lips_samples"


class LipsManifestPipelineTests(unittest.TestCase):
    def test_build_manifest_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = build_lips_manifest(
                LipsBuildConfig(
                    input_root=FIXTURES_DIR,
                    output_dir=Path(tmp_dir),
                    review_sample_size=4,
                    min_candidate_tokens=20,
                )
            )

            included_rows = read_jsonl(report.included_path)
            excluded_rows = read_jsonl(report.excluded_path)
            review_rows = read_jsonl(report.review_sample_path)

        self.assertEqual(report.included_sections, 6)
        self.assertEqual(report.excluded_sections, 3)
        self.assertAlmostEqual(report.parse_success_ratio, 8 / 9, places=4)
        self.assertEqual(sorted(report.counts_by_family), [
            "opinion_monologue",
            "personal_experience",
            "picture_description",
            "travel_narrative",
        ])
        self.assertEqual(len(included_rows), 6)
        self.assertEqual(len(excluded_rows), 3)
        self.assertEqual(len(review_rows), 4)
        self.assertEqual(sorted(report.counts_by_cefr), ["A1", "A2", "B1", "B2", "C1", "C2"])
        self.assertEqual(report.exclusion_reason_counts["raw_mode_dialogue"], 2)
        self.assertEqual(report.exclusion_reason_counts["placeholder_section"], 1)

    def test_validation_requires_manual_review_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = build_lips_manifest(
                LipsBuildConfig(input_root=FIXTURES_DIR, output_dir=Path(tmp_dir), review_sample_size=4)
            )
            validation = validate_lips_manifest(
                report.included_path,
                report.excluded_path,
                config=LipsValidationConfig(
                    min_usable_sections=6,
                    min_task_families=4,
                    target_parse_success_ratio=0.8,
                    min_manual_agreement=0.85,
                ),
            )

        self.assertFalse(validation.hard_gate_passed)
        self.assertFalse(validation.overall_passed)
        self.assertFalse(validation.gate_results["min_manual_agreement"])

    def test_validation_passes_with_completed_review_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = build_lips_manifest(
                LipsBuildConfig(input_root=FIXTURES_DIR, output_dir=Path(tmp_dir), review_sample_size=4)
            )
            review_rows = read_jsonl(report.review_sample_path)
            completed_rows = []
            for row in review_rows:
                row["reviewer_accepts_mapping"] = True
                row["reviewer_task_family"] = row["mapped_task_family"]
                row["reviewer_notes"] = "ok"
                completed_rows.append(row)
            review_path = Path(tmp_dir) / "completed_review.jsonl"
            write_jsonl(review_path, completed_rows)

            validation = validate_lips_manifest(
                report.included_path,
                report.excluded_path,
                review_path=review_path,
                config=LipsValidationConfig(
                    min_usable_sections=6,
                    min_task_families=4,
                    target_parse_success_ratio=0.8,
                    min_manual_agreement=0.85,
                ),
            )

        self.assertTrue(validation.hard_gate_passed)
        self.assertTrue(validation.parse_target_passed)
        self.assertTrue(validation.overall_passed)
        self.assertEqual(validation.manual_reviewed_count, 4)
        self.assertEqual(validation.manual_agreement_ratio, 1.0)

    def test_build_manifest_keeps_monologue_with_light_examiner_scaffolding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "scaffold_B1.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 1",
                        "SE 2 giornata tipica M",
                        "E: Va bene.",
                        (
                            "C: La mia giornata tipica comincia presto e di solito faccio colazione "
                            "con la mia famiglia, poi vado al lavoro, nel pomeriggio torno a casa "
                            "e la sera leggo o faccio una passeggiata nel quartiere."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "personal_experience")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_monologue_with_repeated_examiner_backchannels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "backchannels_B2.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 2",
                        "SE 2 lo sport M",
                        "E: Va bene, puoi iniziare.",
                        (
                            "C: Fare sport per me e molto importante perche mi aiuta a stare bene "
                            "e a organizzare meglio la settimana."
                        ),
                        "E: Si.",
                        (
                            "C: Di solito vado in palestra due volte e nel fine settimana faccio una "
                            "passeggiata lunga oppure vado in bicicletta."
                        ),
                        "E: Capisco.",
                        (
                            "C: Mi piace soprattutto muovermi all'aperto quando il tempo e bello "
                            "perche la campagna mi rilassa."
                        ),
                        "E: Continua.",
                        (
                            "C: Quando posso gioco anche con gli amici e questo rende l'attivita piu "
                            "divertente e meno faticosa."
                        ),
                        "E: Bene.",
                        (
                            "C: Per questo penso che lo sport faccia bene non solo al corpo ma anche "
                            "all'umore e alla vita quotidiana."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "personal_experience")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_long_single_turn_after_examiner_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "long_prompt_C1.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 3",
                        "SE 2 turismo ecologico M",
                        (
                            "E: Vorrei che tu spiegassi con calma se il turismo ecologico ti sembra "
                            "una scelta utile, quali vantaggi puo portare ai territori, che rapporto "
                            "ha con il rispetto della natura, e se nelle tue vacanze personali conta "
                            "di piu il divertimento, la cultura, oppure la protezione dell'ambiente."
                        ),
                        (
                            "C: Penso che il turismo ecologico sia una proposta molto interessante "
                            "perche permette di visitare posti nuovi senza rovinare il paesaggio e "
                            "senza trasformare tutto in un consumo veloce. Quando viaggio preferisco "
                            "sempre camminare, usare i mezzi pubblici, parlare con le persone del "
                            "posto e scegliere piccoli alberghi o case gestite da famiglie locali. "
                            "Per me una vacanza bella non significa soltanto riposare ma capire come "
                            "vive una comunita, quali problemi ambientali affronta e che tipo di "
                            "equilibrio esiste tra turismo, lavoro e tutela del territorio. Se un "
                            "luogo e pieno di rumore, rifiuti e traffico, io non riesco davvero a "
                            "godermi il viaggio. Per questo considero il turismo ecologico non una "
                            "moda ma un modo piu responsabile di essere turisti. Inoltre penso che "
                            "questo approccio aiuti anche le economie locali, perche i visitatori "
                            "restano piu tempo, comprano prodotti del posto e imparano a rispettare "
                            "le abitudini della comunita che li ospita. In questo senso il viaggio "
                            "diventa anche un'esperienza educativa che cambia il comportamento del "
                            "turista una volta tornato a casa."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "opinion_monologue")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_long_single_turn_after_heavy_prompt_with_more_backchannels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "paranormale_C2.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 4",
                        "SE 2 paranormale M",
                        "E: Vorrei che tu riflettessi sul paranormale e sull'astrologia.",
                        "E: Molte persone ci credono ancora oggi.",
                        "E: Pensi che abbiano un valore reale oppure simbolico?",
                        (
                            "C: Per me il paranormale affascina molte persone perche promette risposte "
                            "semplici a domande difficili, ma proprio per questo bisogna parlarne con "
                            "prudenza. Capisco chi cerca conforto nell'astrologia o nelle pratiche "
                            "divinatorie, perche spesso nei momenti di incertezza tutti vorremmo un "
                            "segno che ci rassicuri. Tuttavia credo che il rischio sia quello di "
                            "delegare troppo facilmente le proprie decisioni a qualcuno che interpreta "
                            "simboli o coincidenze. Nella mia esperienza le persone possono trovare "
                            "energia e coraggio anche senza attribuire un valore scientifico a questi "
                            "fenomeni, trattandoli piuttosto come racconti culturali o strumenti "
                            "psicologici. Quando se ne parla pubblicamente, secondo me, e importante "
                            "distinguere tra curiosita personale, tradizione popolare e verita "
                            "dimostrabile, altrimenti si finisce per confondere speranza, suggestione "
                            "e conoscenza. Inoltre credo che il fascino del paranormale dipenda anche "
                            "dal bisogno umano di trovare un ordine narrativo negli eventi casuali, "
                            "ma questo non basta per trasformare un'intuizione in una prova. Quando "
                            "i media presentano queste pratiche come se avessero lo stesso peso della "
                            "ricerca scientifica, secondo me alimentano solo altra confusione. Per "
                            "questo preferisco distinguere nettamente tra immaginazione, rituale, "
                            "cultura popolare e conoscenza verificabile, senza negare che certi "
                            "racconti possano avere un valore emotivo o letterario."
                        ),
                        "E: Va bene.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "opinion_monologue")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_very_long_monologue_with_many_short_backchannels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "relaxation_C1.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 3",
                        "SE 2 moodi per rilassarsi M",
                        "E: Dimmi come ci si rilassa oggi.",
                        (
                            "C: Secondo me molte persone non riescono piu a rilassarsi davvero perche "
                            "portano il lavoro anche dentro la vita privata e finiscono per sentirsi "
                            "sempre reperibili, come se il telefono e il computer non dessero mai "
                            "una vera pausa."
                        ),
                        "E: Si.",
                        (
                            "C: Per questo credo che il tempo libero debba essere difeso con piu "
                            "decisione, scegliendo attivita che aiutano il corpo e la mente e "
                            "restituiscono una sensazione concreta di calma, non soltanto di svago "
                            "superficiale."
                        ),
                        "E: Continua.",
                        (
                            "C: Alcuni preferiscono lo sport, altri il teatro o la musica, ma il punto "
                            "centrale e ritrovare un ritmo umano dopo giornate troppo veloci e troppo "
                            "frammentate da impegni, notifiche e richieste continue."
                        ),
                        "E: Va bene.",
                        (
                            "C: Personalmente penso che camminare, stare con la famiglia e parlare con "
                            "gli amici siano modi molto efficaci per recuperare equilibrio, soprattutto "
                            "quando si riesce a uscire davvero dalla logica della produttivita."
                        ),
                        "E: Certo.",
                        (
                            "C: Anche leggere o andare al cinema puo funzionare, purche non diventi un "
                            "altro obbligo da incastrare in agenda, perche il riposo perde subito il "
                            "suo valore se viene organizzato come una prestazione."
                        ),
                        "E: Mhmh.",
                        (
                            "C: In generale mi sembra importante che la societa non misuri tutto solo "
                            "in produttivita, perche il riposo non e una perdita di tempo ma una "
                            "condizione per vivere meglio e per non consumare le persone fino "
                            "all'esaurimento."
                        ),
                        "E: Capisco.",
                        (
                            "C: Se impariamo a proteggerlo, il tempo libero migliora anche la qualita "
                            "del lavoro, delle relazioni e della salute, e puo diventare uno spazio "
                            "in cui ciascuno recupera attenzione, creativita e senso delle priorita. "
                            "In questo senso rilassarsi non significa fuggire dalle responsabilita, "
                            "ma dare al pensiero il tempo necessario per ritrovare lucidita, "
                            "immaginazione e un rapporto piu sano con le persone che abbiamo intorno."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "opinion_monologue")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_long_prompt_monologue_split_across_two_candidate_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "artist_public_C2.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 4",
                        "SE 2 artista e pubblico M",
                        (
                            "E: Vorrei che tu riflettessi sul rapporto tra artista e pubblico e su "
                            "come questo rapporto sia cambiato negli ultimi decenni."
                        ),
                        (
                            "C: Secondo me oggi esiste una distanza piu forte tra artista e pubblico "
                            "perche il pubblico e diventato piu esigente e piu rapido nel giudicare, "
                            "mentre l'artista si trova spesso a lavorare dentro un sistema molto "
                            "commerciale. Questo rende piu difficile costruire una relazione lenta, "
                            "profonda e davvero condivisa. A volte si pretende dall'artista una "
                            "risposta immediata, quasi un prodotto, invece di un percorso espressivo."
                        ),
                        "E: Continua.",
                        (
                            "C: Per questo mi colpiscono molto i casi in cui un musicista o uno "
                            "scrittore raccontano apertamente la frattura con il proprio pubblico. "
                            "Quando cambia linguaggio o stile, spesso viene punito proprio da chi lo "
                            "seguiva prima. Io penso che il pubblico abbia diritto di criticare, ma "
                            "che l'artista debba conservare una certa liberta interiore; altrimenti "
                            "si crea solo un muro tra chi produce e chi ascolta."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "opinion_monologue")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_manual_reviewed_christmas_plans_monologue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "christmas_plans_B2.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 2",
                        "SE 2 vacanze di Natale M",
                        "E: Parlami delle vacanze di Natale.",
                        (
                            "C: Per le vacanze di Natale arrivera il mio fidanzato e allora faremo "
                            "tutto quello che abbiamo pensato da tempo, prima di tutto comprare i "
                            "regali per la famiglia e scrivere gli auguri agli amici."
                        ),
                        "E: Bene.",
                        (
                            "C: Poi per la festa di Capodanno forse andremo a una serata elegante "
                            "come abbiamo fatto l'anno scorso, perche ci siamo divertiti tantissimo "
                            "e ci piacerebbe ripetere quell'esperienza."
                        ),
                        "E: Capisco.",
                        (
                            "C: Mi piacerebbe anche viaggiare a Natale, magari in Austria o a Parigi, "
                            "ma in generale penso che questo periodo sia soprattutto un momento per "
                            "restare con le persone care e organizzare con calma tutto il tempo libero."
                        ),
                        "E: Continua.",
                        (
                            "C: Se avremo abbastanza soldi faremo anche una piccola gita, ma la cosa "
                            "piu importante per me resta stare con il mio fidanzato e con la famiglia "
                            "senza il ritmo frenetico del lavoro."
                        ),
                        "E: Va bene.",
                        (
                            "C: Per questo le vacanze di Natale mi sembrano sempre un'occasione "
                            "molto personale, affettiva e serena, piu che una semplice festa nel "
                            "calendario."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "personal_experience")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_build_manifest_keeps_manual_reviewed_piazza_picture_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus_dir = Path(tmp_dir) / "corpus"
            corpus_dir.mkdir()
            (corpus_dir / "piazza_picture_B1.txt").write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 1",
                        "SE 2 foto: descrizione piazza",
                        "E: Descrivi questa piazza.",
                        (
                            "C: C'e una edicola nella piazza, c'e gente che passa e un uomo spinge "
                            "un passeggino vicino a un duomo con marmo bianco e nero molto bello."
                        ),
                        "E: Si.",
                        (
                            "C: Si vede anche un campanile alto, un'aiuola con i fiori, un piccolo "
                            "albero e alcune persone che camminano tranquillamente davanti alla "
                            "facciata antica."
                        ),
                        "E: Va bene.",
                        (
                            "C: Mi sembra una piazza italiana molto viva ma anche ordinata, con "
                            "molti dettagli architettonici e un'atmosfera serena da centro storico. "
                            "Sul fondo si intravedono altre case, le finestre aperte e qualche "
                            "persona che guarda la scena dall'ingresso del duomo, mentre la piazza "
                            "resta luminosa e ben curata come nei centri storici che si vedono nelle "
                            "fotografie turistiche. Si notano anche piccoli contrasti tra le zone di "
                            "ombra e di sole, le persone sembrano passeggiare senza fretta e tutto "
                            "l'insieme comunica l'idea di una cittadina accogliente dove il monumento "
                            "principale resta il punto di riferimento della vita quotidiana."
                        ),
                        "E: Grazie.",
                    ]
                ),
                encoding="iso-8859-1",
            )

            report = build_lips_manifest(
                LipsBuildConfig(input_root=corpus_dir, output_dir=Path(tmp_dir) / "artifacts", review_sample_size=2)
            )
            included_rows = read_jsonl(report.included_path)

        self.assertEqual(report.included_sections, 1)
        self.assertEqual(report.excluded_sections, 0)
        self.assertEqual(len(included_rows), 1)
        included_row = included_rows[0]
        self.assertEqual(included_row["mapped_task_family"], "picture_description")
        self.assertTrue(included_row["needs_review"])
        self.assertIsNone(included_row["exclusion_reason"])

    def test_scripts_run_end_to_end_on_fixture_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir) / "artifacts"
            build_script = REPO_ROOT / "scripts" / "build_lips_manifest.py"
            validate_script = REPO_ROOT / "scripts" / "validate_lips_manifest.py"

            build_completed = subprocess.run(
                [
                    sys.executable,
                    str(build_script),
                    str(FIXTURES_DIR),
                    "--output-dir",
                    str(artifact_dir),
                    "--review-sample-size",
                    "4",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if build_completed.returncode != 0:
                self.fail(
                    "build_lips_manifest.py failed\n"
                    f"stdout:\n{build_completed.stdout}\n"
                    f"stderr:\n{build_completed.stderr}"
                )

            build_payload = json.loads(build_completed.stdout)
            review_rows = read_jsonl(build_payload["review_sample_path"])
            for row in review_rows:
                row["reviewer_accepts_mapping"] = True
                row["reviewer_task_family"] = row["mapped_task_family"]
            review_path = artifact_dir / "completed_review.jsonl"
            write_jsonl(review_path, review_rows)

            validate_completed = subprocess.run(
                [
                    sys.executable,
                    str(validate_script),
                    str(artifact_dir),
                    "--review-file",
                    str(review_path),
                    "--min-usable-sections",
                    "6",
                    "--min-task-families",
                    "4",
                    "--target-parse-success-ratio",
                    "0.8",
                    "--min-manual-agreement",
                    "0.85",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if validate_completed.returncode != 0:
                self.fail(
                    "validate_lips_manifest.py failed\n"
                    f"stdout:\n{validate_completed.stdout}\n"
                    f"stderr:\n{validate_completed.stderr}"
                )

            validation_payload = json.loads(validate_completed.stdout)

        self.assertTrue(validation_payload["overall_passed"])
        self.assertEqual(validation_payload["task_family_coverage"], 4)
        self.assertEqual(sorted(validation_payload["counts_by_cefr"]), ["A1", "A2", "B1", "B2", "C1", "C2"])


if __name__ == "__main__":
    unittest.main()
