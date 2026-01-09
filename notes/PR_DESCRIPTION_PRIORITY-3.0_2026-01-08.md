# Pull Request: Security Patch & Documentation Enhancement

**Branch:** `juniper_canopy/feature/dashboard_upgrade/priority-3.0_2026-01-08`  
**Target:** `main`  
**Date:** 2026-01-08  
**Version:** 0.18.1

---

## Summary

This PR delivers a critical security patch addressing a urllib3 decompression bomb vulnerability, along with documentation infrastructure improvements for standardized security release notes.

---

## Changes

### 🔒 Security

- **urllib3 Dependency Update**: Updated from vulnerable version to `>=2.6.3`
  - Addresses decompression bomb vulnerability (CWE-409)
  - Malicious servers could exploit redirect handling to cause excessive resource consumption
  - [Dependabot Alert #2](https://github.com/pcalnon/Juniper/security/dependabot/2)

### 📚 Documentation

- **Security Release Notes v0.15.1-alpha** (`notes/RELEASE_NOTES_v0.15.1-alpha.md`)
  - Complete security advisory documentation
  - Remediation instructions and upgrade guide
  - Follows standardized security release notes format

- **Security Release Notes Template** (`notes/TEMPLATE_SECURITY_RELEASE_NOTES.md`)
  - Reusable template for future security releases
  - Defines required structure with 11 sections
  - Placeholder markers for easy customization

- **AGENTS.md Update**
  - Added "Security Release Notes" section under Documentation File Types
  - References template as required format for all security releases
  - Links to example release notes (v0.14.1-alpha, v0.15.1-alpha)

### 🔧 Configuration

- **Markdownlint Configuration** (`.markdownlint.json`)
  - Updated rules for template file compatibility

---

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `conf/requirements.txt` | Modified | urllib3 pinned to `>=2.6.3` |
| `notes/RELEASE_NOTES_v0.15.1-alpha.md` | Added | Security release documentation |
| `notes/TEMPLATE_SECURITY_RELEASE_NOTES.md` | Added | Reusable security release template |
| `AGENTS.md` | Modified | Added Security Release Notes section |
| `.markdownlint.json` | Modified | Template compatibility rules |

---

## Security Impact

| Attribute | Value |
|-----------|-------|
| **Severity** | High |
| **Vulnerable Package** | urllib3 ≤2.6.2 |
| **Vulnerability** | Decompression bomb (CWE-409) |
| **Attack Vector** | Malicious HTTP redirect responses |
| **Fix** | urllib3 ≥2.6.3 |

---

## Testing

All existing tests pass. No new tests required for this security/documentation update.

| Metric | Result |
|--------|--------|
| Tests Passed | 2247 |
| Tests Skipped | 34 |
| Coverage | 95%+ |

---

## Upgrade Instructions

After merging, users should update their environments:

```bash
pip install --upgrade urllib3>=2.6.3
```

---

## Related Issues

- [Dependabot Security Alert #2](https://github.com/pcalnon/Juniper/security/dependabot/2)

---

## Checklist

- [x] Security vulnerability addressed
- [x] Release notes created
- [x] Template created for future releases
- [x] AGENTS.md updated with documentation standards
- [x] All tests passing
- [x] No breaking changes
