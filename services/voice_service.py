"""
Service vocal - Compatible toutes versions Streamlit.
Détecte automatiquement la version et utilise la meilleure méthode disponible.
"""

import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
import base64
import tempfile
import os
import re
from typing import Optional
from config.settings import settings
from packaging import version


def _streamlit_version() -> str:
    """Retourne la version de Streamlit installée"""
    import streamlit
    return streamlit.__version__


def _has_audio_input() -> bool:
    """Vérifie si st.audio_input() est disponible (>= 1.35.0)"""
    try:
        return version.parse(_streamlit_version()) >= version.parse("1.35.0")
    except Exception:
        return False


class VoiceService:
    """Service vocal compatible toutes versions Streamlit"""

    def __init__(self):
        self.recognizer  = sr.Recognizer()
        self.speech_lang = settings.SPEECH_LANG  # fr-FR
        self.tts_lang    = settings.TTS_LANG      # fr

    # ------------------------------------------------------------------ #
    #  SPEECH TO TEXT
    # ------------------------------------------------------------------ #

    def render_audio_input(self) -> Optional[str]:
        """
        Affiche le widget d'enregistrement adapté à la version Streamlit.

        Returns:
            Texte transcrit ou None
        """
        if _has_audio_input():
            return self._render_audio_input_native()
        else:
            return self._render_audio_input_fallback()

    def _render_audio_input_native(self) -> Optional[str]:
        """
        Méthode moderne : st.audio_input() (Streamlit >= 1.35)
        """
        st.caption(f"Version Streamlit : {_streamlit_version()} ✅")

        audio_data = st.audio_input(         # type: ignore[attr-defined]
            label="Enregistrer ma question",
            key="voice_recorder"
        )

        if audio_data is not None:
            with st.spinner("🔄 Transcription en cours..."):
                return self._transcribe_from_bytes(audio_data.getvalue(), suffix=".wav")

        return None

    def _render_audio_input_fallback(self) -> Optional[str]:
        """
        Méthode alternative : upload d'un fichier audio enregistré.
        Compatible avec toutes les versions de Streamlit.
        """
        st.info(f"""
        ℹ️ **Votre version de Streamlit ({_streamlit_version()}) ne supporte pas 
        l'enregistrement direct.**
        
        **Option 1 — Mettre à jour Streamlit (recommandé) :**
        ```
        pip install --upgrade streamlit
        ```
        Puis relancez l'application.
        
        **Option 2 — Uploader un fichier audio :**
        Enregistrez votre question avec votre application audio 
        (Voice Recorder, Audacity...) et uploadez le fichier ci-dessous.
        """)

        audio_file = st.file_uploader(
            "📂 Uploader un fichier audio",
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            key="audio_file_uploader",
            help="Enregistrez votre question et uploadez le fichier ici"
        )

        if audio_file is not None:
            st.audio(audio_file)

            if st.button("🔄 Transcrire", key="btn_transcribe", type="primary"):
                with st.spinner("🔄 Transcription en cours..."):
                    _ALLOWED_AUDIO_EXT = {'.wav', '.mp3', '.ogg', '.m4a', '.flac'}
                    raw_ext = '.' + audio_file.name.rsplit('.', 1)[-1].lower()
                    suffix = raw_ext if raw_ext in _ALLOWED_AUDIO_EXT else '.wav'
                    return self._transcribe_from_bytes(
                        audio_file.getvalue(),
                        suffix=suffix
                    )

        return None

    def _transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav"
    ) -> Optional[str]:
        """
        Transcrit des bytes audio en texte.

        Args:
            audio_bytes: Données audio brutes
            suffix: Extension du fichier temporaire

        Returns:
            Texte transcrit ou None
        """
        tmp_path = None
        try:
            # Écrire dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Convertir en WAV si nécessaire (mp3, ogg, m4a...)
            wav_path = tmp_path
            if suffix.lower() != ".wav":
                wav_path = self._convert_to_wav(tmp_path)

            # Transcrire avec speech_recognition
            with sr.AudioFile(wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.record(source)

            text = self.recognizer.recognize_google(
                audio,
                language=self.speech_lang
            )
            return text

        except sr.UnknownValueError:
            st.error("❌ Je n'ai pas pu comprendre. Parle plus clairement et réessaie.")
            return None
        except sr.RequestError as e:
            st.error(f"❌ Service de reconnaissance indisponible : {e}")
            return None
        except Exception as e:
            st.error(f"❌ Erreur lors de la transcription : {e}")
            return None
        finally:
            # Nettoyer les fichiers temporaires
            for path in [tmp_path, wav_path if 'wav_path' in locals() else None]:
                if path and path != tmp_path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _convert_to_wav(self, input_path: str) -> str:
        """
        Convertit un fichier audio en WAV via pydub.

        Args:
            input_path: Chemin du fichier source

        Returns:
            Chemin du fichier WAV converti
        """
        try:
            from pydub import AudioSegment

            ext = os.path.splitext(input_path)[1].lower().strip(".")
            audio = AudioSegment.from_file(input_path, format=ext)

            wav_path = input_path.replace(f".{ext}", ".wav")
            audio.export(wav_path, format="wav")
            return wav_path

        except ImportError:
            st.warning("⚠️ pydub non installé. `pip install pydub` pour supporter MP3/OGG.")
            return input_path
        except Exception as e:
            st.warning(f"⚠️ Conversion audio échouée : {e}")
            return input_path

    # ------------------------------------------------------------------ #
    #  TEXT TO SPEECH
    # ------------------------------------------------------------------ #

    def text_to_speech(self, text: str, auto_play: bool = False) -> str:
        """
        Convertit du texte en audio MP3 via gTTS.

        Args:
            text: Texte à synthétiser
            auto_play: Lancer automatiquement la lecture

        Returns:
            Balise HTML <audio> encodée en base64
        """
        try:
            clean_text = self._clean_text_for_speech(text)

            if not clean_text.strip():
                return ""

            tts = gTTS(text=clean_text, lang=self.tts_lang, slow=False)

            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            audio_b64    = base64.b64encode(audio_buffer.read()).decode()
            autoplay_attr = "autoplay" if auto_play else ""

            return f"""
            <audio controls {autoplay_attr} style="width:100%; margin-top:8px;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            </audio>
            """

        except Exception as e:
            st.warning(f"⚠️ Synthèse vocale indisponible : {e}")
            return ""

    def render_tts_button(self, text: str, key: str = "tts_btn"):
        """
        Affiche un bouton 🔊 qui lit le texte à voix haute.

        Args:
            text: Texte à lire
            key: Clé unique Streamlit
        """
        if st.button("🔊 Écouter", key=key, help="Lire la réponse à voix haute"):
            with st.spinner("🎵 Génération audio..."):
                audio_html = self.text_to_speech(text, auto_play=True)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    #  UTILITAIRES
    # ------------------------------------------------------------------ #

    def _clean_text_for_speech(self, text: str) -> str:
        """
        Nettoie le texte avant la synthèse vocale et convertit les
        notations mathématiques en langage naturel français.
        """
        # Supprimer LaTeX block et inline
        text = re.sub(r'\$\$[^$]+\$\$', 'formule mathématique', text)
        text = re.sub(r'\$[^$]+\$', 'formule', text)

        # Supprimer markdown
        text = re.sub(r'[*_#`~>|]', '', text)

        # ── Symboles Unicode → langage naturel ──────────────────────────────
        replacements = [
            # Opérations
            ('×', ' fois '),
            ('÷', ' divisé par '),
            ('±', ' plus ou moins '),
            ('·', ' fois '),
            # Comparaisons
            ('≠', ' différent de '),
            ('≤', ' inférieur ou égal à '),
            ('≥', ' supérieur ou égal à '),
            ('≈', ' environ égal à '),
            ('≡', ' équivalent à '),
            # Flèches / logique
            ('→', ' tend vers '),
            ('⇒', ' implique '),
            ('⟺', ' équivaut à '),
            # Ensembles
            ('∈', ' appartient à '),
            ('∉', ' n\'appartient pas à '),
            ('⊂', ' inclus dans '),
            ('∪', ' union '),
            ('∩', ' intersection '),
            ('∅', ' ensemble vide '),
            ('ℝ', 'R'),
            ('ℕ', 'N'),
            ('ℤ', 'Z'),
            ('ℚ', 'Q'),
            ('ℂ', 'C'),
            # Quantificateurs
            ('∀', 'pour tout '),
            ('∃', 'il existe '),
            # Infini
            ('∞', 'l\'infini'),
            # Exposants Unicode → texte
            ('²', ' au carré'),
            ('³', ' au cube'),
            ('⁴', ' à la puissance 4'),
            ('⁵', ' à la puissance 5'),
            ('⁶', ' à la puissance 6'),
            ('⁷', ' à la puissance 7'),
            ('⁸', ' à la puissance 8'),
            ('⁹', ' à la puissance 9'),
            ('ⁿ', ' à la puissance n'),
            # Indices Unicode → texte
            ('₀', ' zéro'),
            ('₁', ' un'),
            ('₂', ' deux'),
            ('₃', ' trois'),
            ('₄', ' quatre'),
            ('₅', ' cinq'),
            ('₆', ' six'),
            ('₇', ' sept'),
            ('₈', ' huit'),
            ('₉', ' neuf'),
            ('ₙ', ' n'),
            # Racine
            ('√', ' racine de '),
            # Somme / produit / intégrale
            ('∑', ' somme '),
            ('∏', ' produit '),
            ('∫', ' intégrale de '),
            # Lettres grecques
            ('α', 'alpha'),
            ('β', 'bêta'),
            ('γ', 'gamma'),
            ('δ', 'delta'),
            ('Δ', 'delta'),
            ('ε', 'epsilon'),
            ('θ', 'thêta'),
            ('λ', 'lambda'),
            ('μ', 'mu'),
            ('π', 'pi'),
            ('σ', 'sigma'),
            ('ω', 'oméga'),
        ]
        for symbol, word in replacements:
            text = text.replace(symbol, word)

        # f(x) → f de x
        text = re.sub(r'\b([a-zA-Z])\(([a-zA-Z])\)', r'\1 de \2', text)

        # Fractions écrites a/b → a sur b (contexte math : entourées de lettres/chiffres)
        text = re.sub(r'([A-Za-z0-9]+)/([A-Za-z0-9]+)', r'\1 sur \2', text)

        # Séquences de lettres majuscules (noms de points/segments) : ABC → A B C
        def _space_caps(m):
            return ' '.join(list(m.group(0)))
        text = re.sub(r'\b[A-Z]{2,4}\b', _space_caps, text)

        # Supprimer les liens
        text = re.sub(r'http\S+', '', text)

        # Supprimer les caractères non prononçables restants (hors ASCII + accents français)
        text = re.sub(r'[^\x00-\xFF]', '', text)

        # Nettoyer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()

        # Limiter la longueur (gTTS limite)
        if len(text) > 3000:
            text = text[:3000] + "... La réponse a été tronquée."

        return text


# Instance globale
voice_service = VoiceService()
