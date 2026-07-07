param(
    [string]$OutputDir = "outputs",
    [ValidateSet("recent", "legacy")]
    [string]$Lut = "recent"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Drawing

$root = (Get-Location).Path
$partitionDir = Join-Path $root "partition"
$masksDir = Join-Path $root "masks"
$outDir = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Get-XlsxEntryText {
    param(
        [System.IO.Compression.ZipArchive]$Zip,
        [string]$Name
    )
    $entry = $Zip.GetEntry($Name)
    if ($null -eq $entry) { return $null }
    $reader = New-Object System.IO.StreamReader($entry.Open())
    try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
}

function Get-ColIndex {
    param([string]$CellRef)
    $letters = ([regex]::Match($CellRef, "^[A-Z]+")).Value
    $n = 0
    foreach ($ch in $letters.ToCharArray()) {
        $n = ($n * 26) + ([int][char]$ch - [int][char]'A' + 1)
    }
    return $n - 1
}

function Convert-Flag {
    param($Value)
    if ($null -eq $Value) { return 0 }
    $s = [string]$Value
    if ([string]::IsNullOrWhiteSpace($s)) { return 0 }
    $d = 0.0
    if ([double]::TryParse($s, [Globalization.NumberStyles]::Any, [Globalization.CultureInfo]::InvariantCulture, [ref]$d)) {
        return [int]($d -ne 0.0)
    }
    return [int]($s.Trim().ToLowerInvariant() -in @("true", "yes", "si", "sí"))
}

function Read-XlsxRows {
    param([string]$Path)
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        $shared = @()
        $sharedText = Get-XlsxEntryText $zip "xl/sharedStrings.xml"
        if ($sharedText) {
            [xml]$ss = $sharedText
            $nss = New-Object System.Xml.XmlNamespaceManager($ss.NameTable)
            $nss.AddNamespace("x", $ss.DocumentElement.NamespaceURI)
            foreach ($si in $ss.SelectNodes("//x:si", $nss)) {
                $shared += $si.InnerText
            }
        }

        $sheetText = Get-XlsxEntryText $zip "xl/worksheets/sheet1.xml"
        if (-not $sheetText) { return @() }
        [xml]$sheet = $sheetText
        $nsm = New-Object System.Xml.XmlNamespaceManager($sheet.NameTable)
        $nsm.AddNamespace("x", $sheet.DocumentElement.NamespaceURI)

        $rows = $sheet.SelectNodes("//x:sheetData/x:row", $nsm)
        if ($rows.Count -eq 0) { return @() }

        $header = @{}
        foreach ($c in $rows[0].SelectNodes("x:c", $nsm)) {
            $idx = Get-ColIndex $c.r
            $v = $c.SelectSingleNode("x:v", $nsm)
            if ($null -eq $v) { continue }
            $text = if ($c.t -eq "s") { $shared[[int]$v.InnerText] } else { $v.InnerText }
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $header[$text.Trim()] = $idx
            }
        }
        if (-not $header.ContainsKey("image_name")) { return @() }

        $result = New-Object System.Collections.Generic.List[object]
        for ($r = 1; $r -lt $rows.Count; $r++) {
            $values = @{}
            foreach ($c in $rows[$r].SelectNodes("x:c", $nsm)) {
                $idx = Get-ColIndex $c.r
                $v = $c.SelectSingleNode("x:v", $nsm)
                if ($null -eq $v) { continue }
                $text = if ($c.t -eq "s") { $shared[[int]$v.InnerText] } else { $v.InnerText }
                $values[$idx] = $text
            }
            $name = $values[$header["image_name"]]
            if ([string]::IsNullOrWhiteSpace([string]$name)) { continue }
            $g4 = if ($header.ContainsKey("G4")) { Convert-Flag $values[$header["G4"]] } else { 0 }
            $g4c = if ($header.ContainsKey("G4C")) { Convert-Flag $values[$header["G4C"]] } else { 0 }
            $result.Add([pscustomobject]@{
                image_name = ([string]$name).Trim()
                G4 = $g4
                G4C = $g4c
            })
        }
        return $result
    }
    finally {
        $zip.Dispose()
    }
}

function Add-BitmapHistogram {
    param(
        [string]$Path,
        [long[]]$FullHist,
        [long[]]$Gg4Hist,
        [int]$Gg4Min,
        [int]$Gg4MaxExclusive
    )
    $bmp = [System.Drawing.Bitmap]::FromFile($Path)
    try {
        $rect = New-Object System.Drawing.Rectangle 0, 0, $bmp.Width, $bmp.Height
        $clone = New-Object System.Drawing.Bitmap $bmp.Width, $bmp.Height, ([System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
        $g = [System.Drawing.Graphics]::FromImage($clone)
        try { $g.DrawImage($bmp, 0, 0, $bmp.Width, $bmp.Height) } finally { $g.Dispose() }
        try {
            $data = $clone.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
            try {
                $stride = [Math]::Abs($data.Stride)
                $bytes = New-Object byte[] ($stride * $clone.Height)
                [Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
                for ($y = 0; $y -lt $clone.Height; $y++) {
                    $row = $y * $stride
                    for ($x = 0; $x -lt $clone.Width; $x++) {
                        $i = $row + ($x * 3)
                        $b = [int]$bytes[$i]
                        $gch = [int]$bytes[$i + 1]
                        $r = [int]$bytes[$i + 2]
                        $gray = [int][Math]::Round((0.114 * $b) + (0.587 * $gch) + (0.299 * $r))
                        $FullHist[$gray]++
                        if ($gray -ge $Gg4Min -and $gray -lt $Gg4MaxExclusive) {
                            $Gg4Hist[$gray]++
                        }
                    }
                }
            }
            finally {
                $clone.UnlockBits($data)
            }
        }
        finally {
            $clone.Dispose()
        }
    }
    finally {
        $bmp.Dispose()
    }
}

$excelPatterns = @(
    "Validation\*\Train.xlsx",
    "Validation\*\Test.xlsx",
    "Test\Train.xlsx",
    "Test\Test.xlsx"
)

$xlsxFiles = @()
foreach ($pat in $excelPatterns) {
    $xlsxFiles += Get-ChildItem -Path (Join-Path $partitionDir $pat) -File -ErrorAction SilentlyContinue
}

$byName = @{}
foreach ($xp in ($xlsxFiles | Sort-Object FullName)) {
    foreach ($row in Read-XlsxRows $xp.FullName) {
        if (-not $byName.ContainsKey($row.image_name)) {
            $byName[$row.image_name] = $row
        }
    }
}

$g4cNames = @(
    foreach ($kv in $byName.GetEnumerator()) {
        if ($kv.Value.G4 -eq 1 -and $kv.Value.G4C -eq 1) { $kv.Key }
    }
) | Sort-Object

if ($Lut -eq "recent") {
    $gg4Min = 75
    $gg4Max = 175
} else {
    $gg4Min = 85
    $gg4Max = 160
}

$histFull = New-Object long[] 256
$histGg4 = New-Object long[] 256
$ok = 0
$missing = 0
$failed = 0

foreach ($name in $g4cNames) {
    $path = Join-Path $masksDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        $missing++
        continue
    }
    try {
        Add-BitmapHistogram -Path $path -FullHist $histFull -Gg4Hist $histGg4 -Gg4Min $gg4Min -Gg4MaxExclusive $gg4Max
        $ok++
    }
    catch {
        $failed++
        Write-Warning "No se pudo leer $name : $($_.Exception.Message)"
    }
}

$csvPath = Join-Path $outDir "histograma_g4c_gris_${Lut}.csv"
$rowsOut = for ($i = 0; $i -lt 256; $i++) {
    [pscustomobject]@{
        gray = $i
        full_patch_pixels = $histFull[$i]
        gg4_lut_pixels = $histGg4[$i]
    }
}
$rowsOut | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

$totalFull = ($histFull | Measure-Object -Sum).Sum
$totalGg4 = ($histGg4 | Measure-Object -Sum).Sum

function Show-Top {
    param(
        [long[]]$Hist,
        [long]$Total,
        [string]$Title,
        [int]$N = 20
    )
    Write-Host ""
    Write-Host $Title
    Write-Host ("{0,5} {1,16} {2,10}" -f "gray", "pixels", "%")
    Write-Host ("-" * 35)
    $items = for ($i = 0; $i -lt 256; $i++) {
        [pscustomobject]@{ gray = $i; pixels = $Hist[$i] }
    }
    $items |
        Sort-Object pixels -Descending |
        Select-Object -First $N |
        ForEach-Object {
            $pct = if ($Total -gt 0) { 100.0 * [double]$_.pixels / [double]$Total } else { 0.0 }
            Write-Host ("{0,5} {1,16:N0} {2,9:N4}%" -f $_.gray, $_.pixels, $pct)
        }
}

Write-Host "Excel leidos: $($xlsxFiles.Count)"
Write-Host "Parches unicos en partition: $($byName.Count)"
Write-Host "Parches unicos G4=1 y G4C=1: $($g4cNames.Count)"
Write-Host "Mascaras leidas OK: $ok | missing: $missing | fallidas: $failed"
Write-Host "LUT usada: $Lut ; GG4 = [$gg4Min, $gg4Max)"
Write-Host "CSV: $csvPath"
Write-Host "Total pixels en parches G4C completos: $([long]$totalFull)"
Write-Host "Total pixels dentro de rango GG4 LUT: $([long]$totalGg4)"

Show-Top -Hist $histFull -Total ([long]$totalFull) -Title "Top grises en parches G4C completos"
Show-Top -Hist $histGg4 -Total ([long]$totalGg4) -Title "Top grises SOLO pixels GG4 segun LUT"
