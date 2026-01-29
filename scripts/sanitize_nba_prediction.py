from pathlib import Path
p=Path('c:/Users/DM/Desktop/nba/nba_prediction.py')
s=p.read_text(encoding='utf-8', errors='ignore')
# Replace non-printable/control characters (except tab/newline/CR)
clean=''.join(ch for ch in s if (ch=='\n' or ch=='\r' or ch=='\t' or (32<=ord(ch)<=126)))
# Normalize line endings
clean = clean.replace('\r\n','\n')
p.write_text(clean, encoding='utf-8')
print('Sanitized file written')
