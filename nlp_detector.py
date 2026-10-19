
import re


PATTERNS = {
    "Urgent language": [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bact now\b",
        r"\bexpires today\b",
        r"\blast chance\b",
        r"\bwithin \d+ hours?\b",
    ],
    "Threatening or pressure language": [
        r"\baccount (?:will be|has been) (?:blocked|suspended)\b",
        r"\byou will be arrested\b",
        r"\blegal action\b",
        r"\bfinal warning\b",
        r"\baccount terminated\b",
    ],
    "Password, OTP or banking requests": [
        r"\bpassword\b",
        r"\bone[- ]time password\b",
        r"\botp\b",
        r"\bpin number\b",
        r"\bverify your (?:bank )?account\b",
        r"\bbank details\b",
        r"\bcredit card details\b",
        r"\bshare your (?:pin|credentials)\b",
    ],
    "URLs and links": [
        r"https?://[^\s]+",
        r"\bwww\.[^\s]+",
        r"\b[a-z0-9-]+\.(?:com|net|org|info|xyz|top)\b",
    ],
    "Common spam expressions": [
        r"\byou have won\b",
        r"\bclaim your prize\b",
        r"\bcongratulations\b",
        r"\bfree entry\b",
        r"\bcash prize\b",
        r"\bclaim now\b",
        r"\bclick here\b",
        r"\bguaranteed winner\b",
    ],
}


def detect_suspicious_phrases(message: str) -> dict:
    """Return matched warning categories and a heuristic risk level."""

    text = message.lower()
    findings = {}

    for category, patterns in PATTERNS.items():
        matches = []

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                matches.append(match.group(0))

        if matches:
            findings[category] = sorted(set(matches))

    # This is a transparent heuristic, not a calibrated fraud probability.
    score = 0

    for category in findings:
        if category == "Password, OTP or banking requests":
            score += 3
        elif category == "URLs and links":
            score += 2
        else:
            score += 1

    if score >= 4:
        risk = "High"
    elif score >= 2:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "findings": findings,
        "risk_score": score,
        "risk_level": risk,
    }
