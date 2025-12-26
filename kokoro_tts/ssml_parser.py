from typing import List, Dict, Optional, Union, Tuple
import re
from datetime import datetime
import math

try:
    from defusedxml import ElementTree as ET  # type: ignore
except Exception:
    import xml.etree.ElementTree as ET  # type: ignore


class SSParser:
    """Advanced SSML parser with Google TTS-level feature support."""
    
    def __init__(self):
        self.phoneme_patterns = {
            'ipa': r'^[a-zA-Zəɑæɔɛɪʃθðŋɹʒʧʤˈˌːˑ]+$',
            'x-sampa': r'^[a-zA-Z@\{OEVIS TDNJrZtSdZ"%=:]+$'
        }
        
    def _normalize_ws(self, text: str) -> str:
        """Normalize whitespace while preserving intentional spaces."""
        return " ".join(text.split())
    
    def _parse_rate(self, rate: str, base_speed: float = 1.0) -> float:
        """Parse speech rate with extended support."""
        if not rate:
            return base_speed
        
        r = rate.strip().lower()
        mapping = {
            "x-slow": 0.5,
            "slow": 0.75,
            "medium": 1.0,
            "fast": 1.25,
            "x-fast": 1.5,
            "default": 1.0,
        }
        
        if r in mapping:
            return mapping[r]
        
        # Handle relative changes (+20%, -10%)
        if r.startswith(('+', '-')):
            try:
                percent = float(r[1:-1]) if r.endswith('%') else float(r[1:])
                multiplier = 1 + (percent / 100) if r.startswith('+') else 1 - (abs(percent) / 100)
                return max(0.25, min(4.0, base_speed * multiplier))
            except Exception:
                return base_speed
        
        # Handle percentages
        if r.endswith('%'):
            try:
                val = float(r[:-1]) / 100.0
                return max(0.25, min(4.0, base_speed * val))
            except Exception:
                return base_speed
        
        # Handle numeric values
        try:
            val = float(r)
            return max(0.25, min(4.0, val))
        except Exception:
            return base_speed
    
    def _parse_pitch(self, pitch: str, base_pitch: float = 1.0) -> float:
        """Parse pitch adjustments."""
        if not pitch:
            return base_pitch
        
        p = pitch.strip().lower()
        mapping = {
            "x-low": 0.5,
            "low": 0.75,
            "medium": 1.0,
            "high": 1.25,
            "x-high": 1.5,
            "default": 1.0,
        }
        
        if p in mapping:
            return mapping[p]
        
        # Handle semitone adjustments (+2st, -1st)
        if p.endswith('st'):
            try:
                st = float(p[:-2])
                # Convert semitones to multiplier (2^(st/12))
                multiplier = 2 ** (st / 12)
                return max(0.5, min(2.0, base_pitch * multiplier))
            except Exception:
                return base_pitch
        
        # Handle Hz adjustments (+50Hz, -30Hz)
        if p.endswith('hz'):
            try:
                hz = float(p[:-2])
                # Simple linear approximation
                multiplier = 1 + (hz / 100)
                return max(0.5, min(2.0, multiplier))
            except Exception:
                return base_pitch
        
        # Handle percentages
        if p.endswith('%'):
            try:
                val = float(p[:-1]) / 100.0
                return max(0.5, min(2.0, val))
            except Exception:
                return base_pitch
        
        return base_pitch
    
    def _parse_volume(self, volume: str, base_volume: float = 1.0) -> float:
        """Parse volume adjustments."""
        if not volume:
            return base_volume
        
        v = volume.strip().lower()
        mapping = {
            "silent": 0.0,
            "x-soft": 0.25,
            "soft": 0.5,
            "medium": 1.0,
            "loud": 1.5,
            "x-loud": 2.0,
            "default": 1.0,
        }
        
        if v in mapping:
            return mapping[v]
        
        # Handle dB adjustments (+6dB, -3dB)
        if v.endswith('db'):
            try:
                db = float(v[:-2])
                # Convert dB to linear scale
                multiplier = 10 ** (db / 20)
                return max(0.0, min(4.0, base_volume * multiplier))
            except Exception:
                return base_volume
        
        # Handle percentages
        if v.endswith('%'):
            try:
                val = float(v[:-1]) / 100.0
                return max(0.0, min(4.0, val))
            except Exception:
                return base_volume
        
        return base_volume
    
    def _parse_break_time_ms(self, time_attr: Optional[str], strength_attr: Optional[str]) -> int:
        """Parse break durations with extended precision."""
        if time_attr:
            t = time_attr.strip().lower()
            try:
                if t.endswith('ms'):
                    return max(0, int(float(t[:-2])))
                if t.endswith('s'):
                    return max(0, int(float(t[:-1]) * 1000))
                # Bare number as ms
                return max(0, int(float(t)))
            except Exception:
                pass
        
        # Extended strength mapping
        mapping = {
            "none": 0,
            "x-weak": 150,
            "weak": 300,
            "medium": 500,
            "strong": 750,
            "x-strong": 1000,
        }
        
        s = (strength_attr or "").strip().lower()
        return mapping.get(s, 0)
    
    def _parse_say_as(self, text: str, interpret_as: str, format_: Optional[str] = None) -> str:
        """Process say-as tags with extensive format support."""
        interpret_as = interpret_as.lower()
        
        if interpret_as == "characters":
            # Spell out characters; map '.' to spoken 'dot'
            punct_map = {'.': 'dot'}
            chars = list(text.replace(" ", ""))
            spoken = [punct_map.get(c, c) for c in chars]
            return " ".join(spoken)
        
        elif interpret_as == "cardinal":
            try:
                num = int(text.replace(",", ""))
                return self._number_to_words(num)
            except Exception:
                return text
        
        elif interpret_as == "ordinal":
            try:
                num = int(text.replace(",", ""))
                return self._ordinal_to_words(num)
            except Exception:
                return text
        
        elif interpret_as == "telephone":
            # Format phone numbers
            digits = re.sub(r'\D', '', text)
            if len(digits) == 10:
                return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
            elif len(digits) == 11:
                return f"{digits[0]} {digits[1:4]} {digits[4:7]} {digits[7:]}"
            return " ".join(list(digits))
        
        elif interpret_as == "date":
            try:
                if format_:
                    return self._format_date(text, format_)
                return self._auto_format_date(text)
            except Exception:
                return text
        
        elif interpret_as == "time":
            try:
                return self._format_time(text)
            except Exception:
                return text
        
        elif interpret_as == "currency":
            try:
                amount = float(re.sub(r'[^\d.-]', '', text))
                currency = re.sub(r'[\d.,\s]+', '', text).strip()
                return self._format_currency(amount, currency)
            except Exception:
                return text
        
        elif interpret_as == "fraction":
            try:
                parts = text.split('/')
                if len(parts) == 2:
                    num, den = int(parts[0]), int(parts[1])
                    return self._format_fraction(num, den)
            except Exception:
                pass
            return text
        
        return text

    def _format_decimals_in_text(self, text: str) -> str:
        """Convert decimal numbers like 17.8 to 'seventeen point eight'."""
        digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }

        def repl(match: re.Match) -> str:
            sign = match.group(1) or ''
            int_part = match.group(2)
            frac_part = match.group(3)
            int_words = self._number_to_words(int(int_part))
            frac_words = " ".join(digit_words.get(d, d) for d in frac_part)
            prefix = "minus " if sign == '-' else ""
            return f"{prefix}{int_words} point {frac_words}"

        # Match optional sign, integer part, decimal point, fractional digits
        pattern = re.compile(r"(?<![\w-])(-?)(\d+)\.(\d+)(?![\w-])")
        return pattern.sub(repl, text)

    def _format_currency_phrases_in_text(self, text: str) -> str:
        """Convert currency phrases like $4.5 million -> 'four point five million dollars'.
        Supports common symbols and ISO codes (USD, EUR, GBP, JPY, CNY, INR, AUD, CAD, CHF, HKD, SGD, KRW, BRL, RUB, TRY, ZAR, MXN, SEK, NOK, DKK, NZD, AED, SAR, PLN, CZK).
        """
        currency_map = {
            '$': 'dollars', 'USD': 'dollars', 'US$': 'dollars', 'A$': 'Australian dollars', 'C$': 'Canadian dollars',
            '€': 'euros', 'EUR': 'euros',
            '£': 'pounds', 'GBP': 'pounds',
            '¥': 'yen', 'JPY': 'yen',
            'CNY': 'yuan',
            'HKD': 'Hong Kong dollars',
            'SGD': 'Singapore dollars',
            '₹': 'rupees', 'INR': 'rupees',
            '₩': 'won', 'KRW': 'won',
            'R$': 'reais', 'BRL': 'reais',
            '₽': 'rubles', 'RUB': 'rubles',
            '₺': 'lira', 'TRY': 'lira',
            'CHF': 'francs',
            'ZAR': 'rand',
            'MXN': 'pesos',
            'SEK': 'kronor',
            'NOK': 'kroner',
            'DKK': 'kroner',
            'NZD': 'New Zealand dollars',
            'AED': 'dirhams',
            'SAR': 'riyals',
            'PLN': 'zloty',
            'CZK': 'koruna',
        }
        unit_map = {
            'thousand': 'thousand',
            'million': 'million',
            'billion': 'billion',
            'trillion': 'trillion',
            'k': 'thousand',
            'm': 'million',
            'b': 'billion',
            't': 'trillion',
        }

        digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }

        def number_to_words(n: int) -> str:
            return self._number_to_words(n)

        def frac_to_words(frac: str) -> str:
            return " ".join(digit_words.get(d, d) for d in frac)

        # Patterns: symbol or code, amount (with optional commas), optional unit word or shorthand
        # Note: include multi-char symbols like US$, A$, C$, R$ explicitly
        cur_opts = r"US\$|A\$|C\$|R\$|USD|EUR|GBP|JPY|CNY|HKD|SGD|INR|KRW|BRL|RUB|TRY|CHF|ZAR|MXN|SEK|NOK|DKK|NZD|AED|SAR|PLN|CZK|\$|€|£|¥|₹|₩|₽|₺"
        pattern = re.compile(rf"(?i)(?<!\w)({cur_opts})\s*([\d,]+(?:\.\d+)?)\s*(thousand|million|billion|trillion|k|m|b|t)?(?!\w)")

        def repl(m: re.Match) -> str:
            cur = m.group(1)
            amt = m.group(2)
            unit = m.group(3)
            cur_key = cur.upper() if cur.isalpha() else cur
            cur_name = currency_map.get(cur_key, 'dollars' if cur_key in ('$','US$') else cur_key.lower())
            unit_name = unit_map.get(unit.lower(), '') if unit else ''
            # Remove commas from the amount
            amt_clean = amt.replace(',', '')
            if '.' in amt_clean:
                ip, fp = amt_clean.split('.', 1)
                int_words = number_to_words(int(ip))
                frac_words = frac_to_words(fp)
                core = f"{int_words} point {frac_words}"
            else:
                core = number_to_words(int(amt_clean))
            if unit_name:
                return f"{core} {unit_name} {cur_name}"
            else:
                return f"{core} {cur_name}"

        return pattern.sub(repl, text)

    def _format_hhmm_in_text(self, text: str) -> str:
        """Convert time expressions like 2:30 -> 'two thirty'. Does not add AM/PM."""
        def repl(m: re.Match) -> str:
            h = int(m.group(1))
            mm = m.group(2)
            try:
                hour_words = self._number_to_words(h)
                m_val = int(mm)
                if m_val == 0:
                    minute_words = "o'clock"
                elif m_val < 10:
                    minute_words = f"oh {self._number_to_words(m_val)}"
                else:
                    minute_words = self._number_to_words(m_val)
                return f"{hour_words} {minute_words}"
            except Exception:
                return m.group(0)

        # Match simple HH:MM without trailing letters
        pattern = re.compile(r"(?<!\w)(\d{1,2}):(\d{2})(?!\w)")
        return pattern.sub(repl, text)
    
    def _number_to_words(self, n: int) -> str:
        """Convert number to English words."""
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", 
               "sixty", "seventy", "eighty", "ninety"]
        
        if n == 0:
            return "zero"
        
        def convert_below_1000(num):
            if num < 10:
                return ones[num]
            elif num < 20:
                return teens[num - 10]
            elif num < 100:
                return tens[num // 10] + ("-" + ones[num % 10] if num % 10 else "")
            else:
                return ones[num // 100] + " hundred" + (" and " + convert_below_1000(num % 100) if num % 100 else "")
        
        result = ""
        scales = ["", "thousand", "million", "billion"]
        scale_idx = 0
        
        while n > 0:
            chunk = n % 1000
            if chunk > 0:
                chunk_words = convert_below_1000(chunk)
                if scale_idx > 0:
                    chunk_words += " " + scales[scale_idx]
                result = chunk_words + (" " + result if result else "")
            n //= 1000
            scale_idx += 1
        
        return result.strip()
    
    def _ordinal_to_words(self, n: int) -> str:
        """Convert number to ordinal English words."""
        words = self._number_to_words(n)
        
        # Simple ordinal conversion rules
        if words.endswith("one"):
            return words[:-3] + "first"
        elif words.endswith("two"):
            return words[:-3] + "second"
        elif words.endswith("three"):
            return words[:-3] + "third"
        elif words.endswith("ve"):
            return words[:-2] + "fth"
        elif words.endswith("t"):
            return words + "h"
        elif words.endswith("e"):
            return words[:-1] + "th"
        else:
            return words + "th"
    
    def _auto_format_date(self, date_str: str) -> str:
        """Auto-detect and format date."""
        # Try common formats
        formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
            "%B %d, %Y", "%d %B %Y", "%Y%m%d"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%B %d, %Y")
            except ValueError:
                continue
        
        return date_str
    
    def _format_date(self, date_str: str, fmt: str) -> str:
        """Format date according to specified format."""
        try:
            # Map format codes to readable formats
            format_map = {
                "mdy": "%B %d, %Y",
                "dmy": "%d %B %Y",
                "ymd": "%Y %B %d",
                "md": "%B %d",
                "dm": "%d %B",
                "my": "%B %Y",
                "d": "%d",
                "m": "%B",
                "y": "%Y"
            }
            
            if fmt in format_map:
                output_format = format_map[fmt]
            else:
                # Try to parse as custom format
                output_format = fmt.replace('yy', '%Y').replace('mm', '%m').replace('dd', '%d')
            
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime(output_format)
        except Exception:
            return date_str
    
    def _format_time(self, time_str: str) -> str:
        """Format time for speech."""
        try:
            # Try 24-hour format
            if ':' in time_str:
                parts = time_str.split(':')
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                
                if hour == 0:
                    return f"twelve {self._format_minute(minute)} AM"
                elif hour < 12:
                    return f"{self._number_to_words(hour)} {self._format_minute(minute)} AM"
                elif hour == 12:
                    return f"twelve {self._format_minute(minute)} PM"
                else:
                    return f"{self._number_to_words(hour - 12)} {self._format_minute(minute)} PM"
        except Exception:
            pass
        
        return time_str
    
    def _format_minute(self, minute: int) -> str:
        """Format minutes for time."""
        if minute == 0:
            return "o'clock"
        elif minute < 10:
            return f"oh {self._number_to_words(minute)}"
        else:
            return self._number_to_words(minute)
    
    def _format_currency(self, amount: float, currency: str) -> str:
        """Format currency for speech."""
        currency_map = {
            "USD": "dollars", "EUR": "euros", "GBP": "pounds",
            "JPY": "yen", "CNY": "yuan", "INR": "rupees"
        }
        
        currency_name = currency_map.get(currency.upper(), currency)
        
        # Handle whole numbers
        if amount.is_integer():
            amount_str = self._number_to_words(int(amount))
            return f"{amount_str} {currency_name}"
        
        # Handle cents
        dollars = int(amount)
        cents = int(round((amount - dollars) * 100))
        
        if dollars > 0:
            dollars_str = self._number_to_words(dollars)
            cents_str = self._number_to_words(cents)
            return f"{dollars_str} {currency_name} and {cents_str} cents"
        else:
            cents_str = self._number_to_words(cents)
            return f"{cents_str} cents"
    
    def _format_fraction(self, numerator: int, denominator: int) -> str:
        """Format fractions for speech."""
        num_words = self._number_to_words(numerator)
        den_words = self._ordinal_to_words(denominator)
        
        if numerator == 1:
            return f"one {den_words}"
        else:
            return f"{num_words} {den_words}"
    
    def _validate_phoneme(self, ph: str, alphabet: str) -> bool:
        """Validate phoneme notation."""
        pattern = self.phoneme_patterns.get(alphabet.lower())
        if pattern:
            return bool(re.match(pattern, ph))
        return True  # Unknown alphabet, accept anyway
    
    def _parse_emphasis(self, level: str) -> float:
        """Parse emphasis level to intensity."""
        mapping = {
            "strong": 1.5,
            "moderate": 1.2,
            "reduced": 0.8,
            "none": 1.0,
            "default": 1.0
        }
        return mapping.get(level.lower(), 1.0)


class SSMLEnhancedParser:
    """Main enhanced SSML parser with Google TTS-level features."""
    
    def __init__(self):
        self.ssp = SSParser()
    
    def parse_ssml(self, ssml_text: str, default_voice: Optional[str] = None, 
                   default_lang: Optional[str] = None, default_speed: float = 1.0,
                   default_pitch: float = 1.0, default_volume: float = 1.0) -> List[Dict]:
        """Parse SSML into a sequence of actions.
        
        Returns a list of dicts with actions:
          - speak: {"type": "speak", "text": str, "voice": str, "lang": str, 
                   "speed": float, "pitch": float, "volume": float, "emphasis": float}
          - break: {"type": "break", "time_ms": int}
          - audio: {"type": "audio", "src": str, "clip_begin": int, "clip_end": int}
          - mark: {"type": "mark", "name": str}
          - paragraph: {"type": "paragraph_begin"} / {"type": "paragraph_end"}
          - sentence: {"type": "sentence_begin"} / {"type": "sentence_end"}
        """
        actions: List[Dict] = []
        
        # Remove XML declaration if present
        ssml_text = re.sub(r'^<\?xml[^>]*\?>', '', ssml_text.strip())
        
        # Handle non-XML input
        if not ssml_text.startswith('<speak'):
            text = self.ssp._normalize_ws(ssml_text)
            # Apply currency phrase normalization, HH:MM, then decimals
            if text:
                text = self.ssp._format_currency_phrases_in_text(text)
                text = self.ssp._format_hhmm_in_text(text)
                text = self.ssp._format_decimals_in_text(text)
                actions.append({
                    "type": "speak", 
                    "text": text, 
                    "voice": default_voice, 
                    "lang": default_lang, 
                    "speed": default_speed,
                    "pitch": default_pitch,
                    "volume": default_volume,
                    "emphasis": 1.0
                })
            return actions
        
        try:
            # Wrap in root if needed
            if not ssml_text.startswith('<?xml'):
                ssml_text = f'<?xml version="1.0"?>{ssml_text}'
            
            root = ET.fromstring(ssml_text)
        except Exception as e:
            # Fallback to plain text
            text = self.ssp._normalize_ws(ssml_text.replace('<speak>', '').replace('</speak>', ''))
            if text:
                text = self.ssp._format_currency_phrases_in_text(text)
                text = self.ssp._format_hhmm_in_text(text)
                text = self.ssp._format_decimals_in_text(text)
                actions.append({
                    "type": "speak", 
                    "text": text, 
                    "voice": default_voice, 
                    "lang": default_lang, 
                    "speed": default_speed,
                    "pitch": default_pitch,
                    "volume": default_volume,
                    "emphasis": 1.0
                })
            return actions
        
        # Parse XML namespace
        namespace = {'ssml': 'http://www.w3.org/2001/10/synthesis'}
        
        # Extract default lang from speak tag
        speak_lang = root.get('{http://www.w3.org/XML/1998/namespace}lang') or \
                    root.get('xml:lang') or root.get('lang')
        
        # Context stack for nested elements
        context_stack = [{
            "voice": default_voice,
            "lang": speak_lang or default_lang,
            "speed": default_speed,
            "pitch": default_pitch,
            "volume": default_volume,
            "emphasis": 1.0,
            "phoneme": None,
            "say_as": None,
            "say_as_format": None,
            "sub_alias": None
        }]
        
        def current_context():
            return context_stack[-1]
        
        def push_context(**updates):
            new_ctx = current_context().copy()
            new_ctx.update(updates)
            context_stack.append(new_ctx)
        
        def pop_context():
            if len(context_stack) > 1:
                context_stack.pop()
        
        def process_text(text: str):
            """Process text with current context."""
            if not text:
                return
            
            text = self.ssp._normalize_ws(text)
            if not text:
                return
            
            ctx = current_context()
            
            # Apply phoneme replacement if active
            if ctx["phoneme"]:
                ph, alphabet = ctx["phoneme"]
                if self.ssp._validate_phoneme(ph, alphabet):
                    # For Kokoro, you might handle this differently
                    text = f"[PHONEME:{alphabet}:{ph}]"
            
            # Apply say-as interpretation
            if ctx["say_as"]:
                text = self.ssp._parse_say_as(text, ctx["say_as"], ctx["say_as_format"])

            # Convert currency phrases (e.g., $4.5 million -> 'four point five million dollars') before other numeric formats
            text = self.ssp._format_currency_phrases_in_text(text)

            # Convert HH:MM times (e.g., 2:30 -> 'two thirty')
            text = self.ssp._format_hhmm_in_text(text)

            # Convert decimals (e.g., 17.8 -> 'seventeen point eight')
            text = self.ssp._format_decimals_in_text(text)
            
            # Apply sub alias
            if ctx["sub_alias"]:
                text = ctx["sub_alias"]
            
            if text:
                actions.append({
                    "type": "speak",
                    "text": text,
                    "voice": ctx["voice"],
                    "lang": ctx["lang"],
                    "speed": ctx["speed"],
                    "pitch": ctx["pitch"],
                    "volume": ctx["volume"],
                    "emphasis": ctx["emphasis"]
                })
        
        def process_element(elem, in_paragraph=False, in_sentence=False):
            """Recursively process SSML elements."""
            tag = elem.tag
            if '}' in tag:
                tag = tag.split('}', 1)[1]  # Remove namespace
            
            # Handle paragraph and sentence boundaries
            if tag == 'p' and not in_paragraph:
                actions.append({"type": "paragraph_begin"})
                in_paragraph = True
            elif tag == 's' and not in_sentence:
                actions.append({"type": "sentence_begin"})
                in_sentence = True
            
            # For context-altering tags, push context before processing inner text
            context_altering = {'voice', 'prosody', 'emphasis', 'say-as', 'phoneme', 'sub'}
            process_inner_text_first = tag in context_altering
            
            # Handle element-specific processing
            if tag == 'break':
                time_ms = self.ssp._parse_break_time_ms(
                    elem.get('time'), 
                    elem.get('strength')
                )
                actions.append({"type": "break", "time_ms": time_ms})
            
            elif tag == 'voice':
                push_context(
                    voice=elem.get('name') or current_context()["voice"],
                    lang=elem.get('{http://www.w3.org/XML/1998/namespace}lang') or 
                         elem.get('xml:lang') or elem.get('lang') or current_context()["lang"]
                )
                if elem.text:
                    process_text(elem.text)
            
            elif tag == 'prosody':
                push_context(
                    speed=self.ssp._parse_rate(elem.get('rate', ''), current_context()["speed"]),
                    pitch=self.ssp._parse_pitch(elem.get('pitch', ''), current_context()["pitch"]),
                    volume=self.ssp._parse_volume(elem.get('volume', ''), current_context()["volume"])
                )
                if elem.text:
                    process_text(elem.text)
            
            elif tag == 'emphasis':
                emphasis_level = self.ssp._parse_emphasis(elem.get('level', 'moderate'))
                push_context(emphasis=emphasis_level)
                if elem.text:
                    process_text(elem.text)
            
            elif tag == 'say-as':
                interpret_as = elem.get('interpret-as', '')
                format_ = elem.get('format')
                push_context(say_as=interpret_as, say_as_format=format_)
                # Text inside say-as will be processed in process_text
                if elem.text:
                    process_text(elem.text)
            
            elif tag == 'phoneme':
                ph = elem.get('ph', '')
                alphabet = elem.get('alphabet', 'ipa').lower()
                if ph and alphabet in ['ipa', 'x-sampa']:
                    push_context(phoneme=(ph, alphabet))
                    if elem.text:
                        process_text(elem.text)
            
            elif tag == 'sub':
                alias = elem.get('alias', '')
                if alias:
                    push_context(sub_alias=alias)
                    if elem.text:
                        process_text(elem.text)
            
            elif tag == 'audio':
                src = elem.get('src', '')
                if src:
                    clip_begin = elem.get('clipBegin')
                    clip_end = elem.get('clipEnd')
                    actions.append({
                        "type": "audio",
                        "src": src,
                        "clip_begin": int(float(clip_begin)) if clip_begin else None,
                        "clip_end": int(float(clip_end)) if clip_end else None
                    })
                # Audio tag does not alter text context; process its inner text normally if present
                if elem.text and not process_inner_text_first:
                    process_text(elem.text)
            
            elif tag == 'mark':
                name = elem.get('name', '')
                if name:
                    actions.append({"type": "mark", "name": name})
                if elem.text and not process_inner_text_first:
                    process_text(elem.text)
            else:
                # Non context-altering tag; process inner text now
                if elem.text and not process_inner_text_first:
                    process_text(elem.text)
            
            # Process child elements
            for child in elem:
                process_element(child, in_paragraph, in_sentence)
            
            # Handle closing of context-altering tags
            if tag in ['voice', 'prosody', 'emphasis', 'say-as', 'phoneme', 'sub']:
                pop_context()
            
            # Process text after this element
            if elem.tail:
                process_text(elem.tail)
            
            # Close paragraph and sentence boundaries
            if tag == 'p' and in_paragraph:
                actions.append({"type": "paragraph_end"})
                in_paragraph = False
            elif tag == 's' and in_sentence:
                actions.append({"type": "sentence_end"})
                in_sentence = False
        
        # Start processing from root
        process_element(root)
        
        return actions


# For backward compatibility
def parse_ssml(ssml_text: str, default_voice: Optional[str] = None, 
               default_lang: Optional[str] = None, default_speed: float = 1.0) -> List[Dict]:
    """Legacy interface for backward compatibility."""
    parser = SSMLEnhancedParser()
    return parser.parse_ssml(ssml_text, default_voice, default_lang, default_speed)


# Example usage with enhanced features
if __name__ == "__main__":
    parser = SSMLEnhancedParser()
    
    ssml_example = """
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
        <p>
            <s>Your account balance is <say-as interpret-as="currency" format="USD">$1,234.56</say-as>.</s>
            <s>The meeting is on <say-as interpret-as="date" format="mdy">12/25/2023</say-as>.</s>
        </p>
        <break time="500ms"/>
        <prosody rate="fast" pitch="+1st" volume="+6dB">
            <emphasis level="strong">Important announcement!</emphasis>
        </prosody>
        <voice name="en-US-Wavenet-A">
            <phoneme alphabet="ipa" ph="həˈloʊ">hello</phoneme> world!
        </voice>
    </speak>
    """
    
    result = parser.parse_ssml(ssml_example)
    for action in result:
        print(action)