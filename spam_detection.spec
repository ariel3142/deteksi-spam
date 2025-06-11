# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['spam_detection.py'],
    pathex=[],
    binaries=[],
    datas=[
        (r'C:\Users\LENOVO\projek akhir data science\venv\Lib\site-packages\Sastrawi\Stemmer\data\kata-dasar.txt', r'Sastrawi\Stemmer\data'),
    ],
    hiddenimports=[
        'Sastrawi',
        'Sastrawi.Stemmer',
        'Sastrawi.Stemmer.StemmerFactory',
        'nltk',
        'nltk.corpus',
        'nltk.corpus.stopwords'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='spam_detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
