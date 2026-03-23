from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from corpora.lips_dataset import _map_task_family, default_lips_output_dir, parse_lips_file


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "corpora" / "lips_samples"


class LipsDatasetTests(unittest.TestCase):
    def test_default_lips_output_dir_stays_at_repo_tmp_root(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "tmp" / "lips_manifest"
        self.assertEqual(default_lips_output_dir(), expected)

    def test_parse_lips_file_reads_dialogue_and_monologue_sections(self) -> None:
        parsed = parse_lips_file(FIXTURES_DIR / "fixture_b1_B1.txt")

        self.assertEqual(parsed.source_metadata.site, "Citta del Messico")
        self.assertEqual(parsed.source_metadata.candidate_id, "13148C")
        self.assertEqual(len(parsed.sections), 2)
        self.assertFalse(parsed.failures)

        section_one, section_two = parsed.sections
        self.assertEqual(section_one.section_id, "SE1")
        self.assertEqual(section_one.raw_mode, "D")
        self.assertEqual(section_one.turn_structure_flag, "dialogue_like")
        self.assertEqual(section_one.candidate_turn_count, 2)
        self.assertEqual(section_one.examiner_turn_count, 2)
        self.assertIn("Verona", section_one.candidate_text_clean)

        self.assertEqual(section_two.section_id, "SE2")
        self.assertEqual(section_two.raw_mode, "M")
        self.assertEqual(section_two.turn_structure_flag, "monologue_like")
        self.assertEqual(section_two.candidate_turn_count, 1)
        self.assertEqual(section_two.examiner_turn_count, 1)
        self.assertEqual(section_two.cefr_level, "B1")
        self.assertNotIn("E:", section_two.candidate_text_clean)
        self.assertGreaterEqual(section_two.candidate_token_count, 20)

    def test_parse_lips_file_supports_indexed_and_lowercase_turn_markers(self) -> None:
        parsed = parse_lips_file(FIXTURES_DIR / "fixture_c1_C1.txt")

        self.assertEqual(len(parsed.sections), 1)
        section = parsed.sections[0]
        self.assertEqual(section.raw_mode, "M")
        self.assertEqual(section.turn_structure_flag, "monologue_like")
        self.assertEqual(section.examiner_turn_count, 1)
        self.assertEqual(section.candidate_turn_count, 1)
        self.assertEqual(section.cefr_level, "C1")
        self.assertIn("viaggio", section.prompt_topic.casefold())

    def test_parse_lips_file_covers_lower_levels(self) -> None:
        parsed_a1 = parse_lips_file(FIXTURES_DIR / "fixture_a1_A1.txt")
        parsed_a2 = parse_lips_file(FIXTURES_DIR / "fixture_a2_A2.txt")

        self.assertEqual(parsed_a1.sections[0].cefr_level, "A1")
        self.assertEqual(parsed_a2.sections[0].cefr_level, "A2")
        self.assertEqual(parsed_a1.sections[0].turn_structure_flag, "monologue_like")
        self.assertEqual(parsed_a2.sections[0].turn_structure_flag, "monologue_like")
        self.assertGreaterEqual(parsed_a1.sections[0].candidate_token_count, 20)
        self.assertGreaterEqual(parsed_a2.sections[0].candidate_token_count, 20)

    def test_parse_lips_file_normalizes_spaced_section_headers_and_mixed_modes(self) -> None:
        parsed = parse_lips_file(FIXTURES_DIR / "fixture_mixed_C1.txt")

        self.assertEqual(parsed.source_metadata.exam_date, "06/12/2004")
        self.assertEqual(parsed.source_metadata.raw_level, "3")
        self.assertEqual(len(parsed.sections), 2)

        mixed, placeholder = parsed.sections
        self.assertEqual(mixed.section_id, "SE1")
        self.assertEqual(mixed.raw_mode, "DM")
        self.assertEqual(mixed.turn_structure_flag, "dialogue_like")
        self.assertEqual(mixed.candidate_turn_count, 2)

        self.assertEqual(placeholder.section_id, "SE2")
        self.assertEqual(placeholder.parse_status, "text_only")
        self.assertEqual(placeholder.exclusion_reason, "placeholder_section")

    def test_parse_lips_file_reports_missing_section_marker_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            broken_path = Path(tmp_dir) / "broken_B1.txt"
            broken_path.write_text("Data esame: 01/01/2001\nnessuna sezione qui\n", encoding="utf-8")

            parsed = parse_lips_file(broken_path)

        self.assertEqual(len(parsed.sections), 0)
        self.assertEqual(len(parsed.failures), 1)
        self.assertEqual(parsed.failures[0].exclusion_reason, "missing_section_marker")

    def test_parse_lips_file_keeps_blank_topic_monologue_as_partial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "blank_topic_B2.txt"
            source_path.write_text(
                "\n".join(
                    [
                        "Data esame: 01/01/2001",
                        "Livello 2",
                        "SE 2 M",
                        "E: Parla pure.",
                        (
                            "C: Vorrei vivere vicino al lago perche mi piace la natura e mi piace "
                            "camminare in montagna durante il fine settimana."
                        ),
                    ]
                ),
                encoding="iso-8859-1",
            )

            parsed = parse_lips_file(source_path)

        self.assertEqual(len(parsed.sections), 1)
        section = parsed.sections[0]
        self.assertIsNone(section.prompt_topic)
        self.assertEqual(section.parse_status, "partial_metadata")
        self.assertIsNone(section.exclusion_reason)
        self.assertEqual(section.turn_structure_flag, "monologue_like")
        self.assertGreater(section.candidate_token_count, 10)

    def test_map_task_family_covers_low_confidence_review_topics(self) -> None:
        expected_mappings = {
            "differenze tra la lingua del candidato e quella italiana": "personal_experience",
            "amici": "personal_experience",
            "aspetto fisico": "personal_experience",
            "città o campagna": "personal_experience",
            "cucina tipica": "personal_experience",
            "il luogo che ami di più": "personal_experience",
            "lingua italiana": "personal_experience",
            "motivo per il quale è in Italia": "personal_experience",
            "paese di origine": "personal_experience",
            "insegnante": "personal_experience",
            "L'Italia e la lingua italiana": "personal_experience",
            "giornata tipica": "personal_experience",
            "vita ideale": "personal_experience",
            "intervista a personaggio famoso": "personal_experience",
            "rapporti con gli Italiani": "personal_experience",
            "descrivere se stessi": "personal_experience",
            "descrizione personale": "personal_experience",
            "la persona più importante nella propria vita": "personal_experience",
            "letture preferite": "personal_experience",
            "una persona cara": "personal_experience",
            "feste tradizionali": "personal_experience",
            "tradizioni popolari del paese": "personal_experience",
            "abitudini alimentari del paese nativo": "personal_experience",
            "università brasiliana": "personal_experience",
            "vacanze di Natale": "personal_experience",
            "proprio padre": "personal_experience",
            "soggiorno in Italia": "personal_experience",
            "la tua città e il tuo paese": "personal_experience",
            "esami della vita": "personal_experience",
            "l'auto condivisa": "opinion_monologue",
            "aspetti dell'Italia": "opinion_monologue",
            "Italia": "opinion_monologue",
            "artista e pubblico": "opinion_monologue",
            "beni culturali del proprio paese": "opinion_monologue",
            "conoscenza lingue": "opinion_monologue",
            "il caldo in Italia": "opinion_monologue",
            "la cultura italiana": "opinion_monologue",
            "la musica e lo sport": "opinion_monologue",
            "le letture preferite": "opinion_monologue",
            "buoni motivi per viaggiare in Italia": "opinion_monologue",
            "le lingue": "opinion_monologue",
            "creatività nella scuola": "opinion_monologue",
            "cura del corpo": "opinion_monologue",
            "libri e 11 settembre, Fallaci": "opinion_monologue",
            "gli Italiani": "opinion_monologue",
            "moodi per rilassarsi": "opinion_monologue",
            "paranormale": "opinion_monologue",
            "donne single": "opinion_monologue",
            "possibilità di svolgere un lavoro part time": "opinion_monologue",
            "passaggio millennio": "opinion_monologue",
            "personaggio pubblico": "opinion_monologue",
            "rapporto tra italiani e stranieri": "opinion_monologue",
            "integrazione interculturale nelle scuole": "opinion_monologue",
            "reality shows": "opinion_monologue",
            "turismo ecologico": "opinion_monologue",
        }

        for topic, expected_family in expected_mappings.items():
            with self.subTest(topic=topic):
                mapped_family, mapping_source, mapping_confidence = _map_task_family(topic)
                self.assertEqual(mapped_family, expected_family)
                self.assertEqual(mapping_source, "heuristic_v2")
                self.assertEqual(mapping_confidence, "medium")

    def test_map_task_family_prefers_topic_signal_over_generic_opinion_words(self) -> None:
        travel_mapping = _map_task_family(
            "vacanze",
            candidate_text="penso che una vacanza lunga sia meglio per vedere posti nuovi",
        )
        personal_mapping = _map_task_family(
            "tempo libero",
            candidate_text="penso che leggere e andare al cinema mi rilassi molto",
        )

        self.assertEqual(travel_mapping, ("travel_narrative", "heuristic_v1", "medium"))
        self.assertEqual(personal_mapping, ("personal_experience", "heuristic_v1", "medium"))

    def test_map_task_family_uses_candidate_image_cues_when_topic_is_ambiguous(self) -> None:
        mapped = _map_task_family(
            "cinema italiano",
            candidate_text="quando ho visto questa immagine mi ha fatto pensare a un vecchio film",
        )

        self.assertEqual(mapped, ("picture_description", "heuristic_v2", "medium"))

    def test_map_task_family_ignores_abstract_image_language(self) -> None:
        mapped = _map_task_family(
            "rapporto tra italiani e stranieri",
            candidate_text="posso parlare dell'immagine che abbiamo noi spagnoli degli italiani",
        )

        self.assertEqual(mapped, ("opinion_monologue", "heuristic_v2", "medium"))

    def test_map_task_family_uses_candidate_future_plan_cues_when_topic_is_missing(self) -> None:
        mapped = _map_task_family(
            None,
            candidate_text=(
                "vorrei fare un corso di guida turistica e poi vorrei lavorare con i visitatori "
                "per spiegare la storia del mio paese"
            ),
        )

        self.assertEqual(mapped, ("personal_experience", "heuristic_v2", "medium"))

    def test_map_task_family_routes_giorni_di_festa_by_candidate_style(self) -> None:
        personal = _map_task_family(
            "giorni di festa",
            candidate_text="per me mi piace svegliarmi con calma e trascorro il tempo con gli amici",
        )
        opinion = _map_task_family(
            "giorni di festa",
            candidate_text="secondo me la festa e il giorno in cui uno puo rilassarsi con la famiglia",
        )

        self.assertEqual(personal, ("personal_experience", "heuristic_v2", "medium"))
        self.assertEqual(opinion, ("opinion_monologue", "heuristic_v2", "medium"))


if __name__ == "__main__":
    unittest.main()
