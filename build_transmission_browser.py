import argparse
import json
import os
import re

import numpy as np


def load_transmissions(npz_path):
    data = np.load(npz_path)

    if "wavelengths" not in data.files:
        raise KeyError("NPZ file must contain 'wavelengths'")

    wavelengths = np.asarray(data["wavelengths"], dtype=float)
    if wavelengths.size == 0:
        raise ValueError("wavelength array is empty")

    tau_keys = [key for key in data.files if key.endswith("__tau")]
    if len(tau_keys) == 0:
        raise KeyError("No ion optical-depth arrays found (expected keys ending with '__tau')")

    total_tau = np.zeros_like(wavelengths, dtype=float)
    ion_tau = {}
    for key in tau_keys:
        tau = np.asarray(data[key], dtype=float)
        if tau.shape != total_tau.shape:
            raise ValueError(f"Shape mismatch for {key}: {tau.shape} != {total_tau.shape}")
        total_tau += tau
        ion_tau[key.removesuffix("__tau")] = tau

    full_transmission = np.exp(-total_tau)
    energy_kev = 12.398419843320026 / wavelengths

    order = np.argsort(energy_kev)
    sorted_energy = energy_kev[order]

    ion_transmissions = {}
    for ion_tag, tau in ion_tau.items():
        ion_transmissions[ion_tag] = np.exp(-tau)[order]

    return sorted_energy, full_transmission[order], ion_transmissions


def ion_display_name(ion_tag):
    parts = ion_tag.split("__")
    return " ".join(part.replace("_", " ") for part in parts if part)


def ion_filename(ion_tag):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", ion_tag).strip("_")
    safe = re.sub(r"_+", "_", safe).lower()
    # Keeping requested filename pattern from the prompt.
    return f"{safe}_transmission.html"


def write_html(out_html, html):
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def build_html(energy_kev, transmission, title, window_kev, step_fraction):
    energy_json = json.dumps(energy_kev.tolist())
    transmission_json = json.dumps(transmission.tolist())

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #14161a;
      --muted: #596273;
      --accent: #0f766e;
      --line: #d7dbe2;
    }}
    body {{
      margin: 0;
      background: radial-gradient(1200px 500px at 10% -10%, #e3f2ef 0%, var(--bg) 40%);
      color: var(--ink);
      font-family: "Source Sans 3", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1300px;
      margin: 22px auto;
      padding: 0 16px 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(20, 22, 26, 0.06);
      overflow: hidden;
    }}
    .head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 8px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 130px 130px 110px 110px;
      gap: 10px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }}
    .controls label {{
      color: var(--muted);
      font-size: 12px;
      display: block;
      margin-bottom: 4px;
    }}
    .controls input[type=number] {{
      width: 100%;
      box-sizing: border-box;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 14px;
    }}
    .controls input[type=range] {{
      width: 100%;
    }}
    .btnrow {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
    }}
    button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    #plot {{
      height: 72vh;
      min-height: 420px;
    }}
    .footer {{
      color: var(--muted);
      font-size: 13px;
      padding: 10px 16px 14px;
      border-top: 1px solid var(--line);
    }}
    @media (max-width: 900px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
      .btnrow {{
        justify-content: flex-start;
      }}
      #plot {{
        height: 64vh;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"panel\">
      <div class=\"head\">
        <div class=\"title\">{title}</div>
        <div class=\"subtitle\">Scroll the full transmission spectrum left to right in your browser. Mouse wheel zooms window size.</div>
      </div>

      <div class=\"controls\">
        <div>
          <label for=\"leftSlider\">Left edge [keV]</label>
          <input id=\"leftSlider\" type=\"range\" />
        </div>
        <div>
          <label for=\"windowInput\">Window [keV]</label>
          <input id=\"windowInput\" type=\"number\" step=\"0.001\" min=\"0.0001\" value=\"{window_kev}\" />
        </div>
        <div>
          <label for=\"stepInput\">Step fraction</label>
          <input id=\"stepInput\" type=\"number\" step=\"0.01\" min=\"0.01\" max=\"1\" value=\"{step_fraction}\" />
        </div>
        <div class=\"btnrow\">
          <button id=\"prevBtn\">Prev</button>
          <button id=\"nextBtn\">Next</button>
        </div>
        <div class=\"btnrow\">
          <button id=\"homeBtn\">Home</button>
          <button class=\"primary\" id=\"endBtn\">End</button>
        </div>
      </div>

      <div id=\"plot\"></div>
      <div class=\"footer\" id=\"status\"></div>
    </div>
  </div>

  <script>
    const energy = {energy_json};
    const transmission = {transmission_json};

    const eMin = energy[0];
    const eMax = energy[energy.length - 1];

    const leftSlider = document.getElementById('leftSlider');
    const windowInput = document.getElementById('windowInput');
    const stepInput = document.getElementById('stepInput');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const homeBtn = document.getElementById('homeBtn');
    const endBtn = document.getElementById('endBtn');
    const status = document.getElementById('status');

    let windowKeV = Math.min(Math.max(parseFloat(windowInput.value), 1e-6), eMax - eMin);
    let stepFraction = Math.min(Math.max(parseFloat(stepInput.value), 0.01), 1.0);
    let left = eMin;

    function clampLeft(x) {{
      return Math.min(Math.max(x, eMin), eMax - windowKeV);
    }}

    function upperBound(arr, x) {{
      let lo = 0;
      let hi = arr.length;
      while (lo < hi) {{
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= x) lo = mid + 1;
        else hi = mid;
      }}
      return lo;
    }}

    function sliceWindow(x0, x1) {{
      const i0 = Math.max(0, upperBound(energy, x0) - 1);
      const i1 = Math.min(energy.length, upperBound(energy, x1) + 1);
      return {{
        x: energy.slice(i0, i1),
        y: transmission.slice(i0, i1),
      }};
    }}

    function updateSliderRange() {{
      leftSlider.min = String(eMin);
      leftSlider.max = String(Math.max(eMin, eMax - windowKeV));
      leftSlider.step = String(Math.max(windowKeV / 300, 1e-6));
      leftSlider.value = String(clampLeft(left));
    }}

    function updatePlot() {{
      left = clampLeft(left);
      const right = Math.min(left + windowKeV, eMax);
      const win = sliceWindow(left, right);

      Plotly.react('plot', [
        {{
          x: win.x,
          y: win.y,
          mode: 'lines',
          line: {{ color: '#c62828', width: 1.6 }},
          name: 'Transmission',
          hovertemplate: 'E=%{{x:.6f}} keV<br>T=%{{y:.6f}}<extra></extra>',
        }}
      ], {{
        margin: {{ l: 70, r: 20, t: 30, b: 60 }},
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: {{ title: 'Energy [keV]', range: [left, right], showgrid: true, gridcolor: '#eef1f4' }},
        yaxis: {{ title: 'exp(-tau_total)', range: [0, 1.02], showgrid: true, gridcolor: '#eef1f4' }},
      }}, {{responsive: true, displaylogo: false}});

      updateSliderRange();
      status.textContent = `Range: ${{left.toFixed(6)}} - ${{right.toFixed(6)}} keV | Window: ${{windowKeV.toFixed(6)}} keV | Step: ${{(windowKeV * stepFraction).toFixed(6)}} keV | Points: ${{win.x.length}}`;
    }}

    function applyInputs() {{
      const w = parseFloat(windowInput.value);
      const s = parseFloat(stepInput.value);

      if (!Number.isFinite(w) || !Number.isFinite(s)) return;

      windowKeV = Math.min(Math.max(w, 1e-6), eMax - eMin);
      stepFraction = Math.min(Math.max(s, 0.01), 1.0);
      windowInput.value = windowKeV.toString();
      stepInput.value = stepFraction.toString();
      left = clampLeft(left);
      updatePlot();
    }}

    leftSlider.addEventListener('input', () => {{
      left = parseFloat(leftSlider.value);
      updatePlot();
    }});

    windowInput.addEventListener('change', applyInputs);
    stepInput.addEventListener('change', applyInputs);

    prevBtn.addEventListener('click', () => {{
      left = clampLeft(left - windowKeV * stepFraction);
      updatePlot();
    }});

    nextBtn.addEventListener('click', () => {{
      left = clampLeft(left + windowKeV * stepFraction);
      updatePlot();
    }});

    homeBtn.addEventListener('click', () => {{
      left = eMin;
      updatePlot();
    }});

    endBtn.addEventListener('click', () => {{
      left = eMax - windowKeV;
      updatePlot();
    }});

    const plotDiv = document.getElementById('plot');
    plotDiv.addEventListener('wheel', (event) => {{
      event.preventDefault();

      const oldWindow = windowKeV;
      const center = left + 0.5 * oldWindow;
      if (event.deltaY < 0) {{
        windowKeV = Math.max(oldWindow * 0.9, 1e-6);
      }} else {{
        windowKeV = Math.min(oldWindow * 1.1, eMax - eMin);
      }}

      left = clampLeft(center - 0.5 * windowKeV);
      windowInput.value = windowKeV.toString();
      updatePlot();
    }}, {{ passive: false }});

    window.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') {{
        left = clampLeft(left - windowKeV * stepFraction);
        updatePlot();
      }} else if (event.key === 'ArrowRight') {{
        left = clampLeft(left + windowKeV * stepFraction);
        updatePlot();
      }} else if (event.key === '+') {{
        windowKeV = Math.max(windowKeV * 0.8, 1e-6);
        windowInput.value = windowKeV.toString();
        updatePlot();
      }} else if (event.key === '-') {{
        windowKeV = Math.min(windowKeV * 1.25, eMax - eMin);
        windowInput.value = windowKeV.toString();
        updatePlot();
      }}
    }});

    updatePlot();
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a browser-based transmission spectrum viewer from long_spectra NPZ"
    )
    parser.add_argument("--npz", required=True, help="Path to long_spectra NPZ output")
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML file path (default: same directory as NPZ, named full_transmission.html)",
    )
    parser.add_argument(
        "--window-kev",
        type=float,
        default=0.08,
        help="Initial displayed energy window in keV",
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=0.15,
        help="Fraction of window moved per step",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    energy_kev, full_transmission, ion_transmissions = load_transmissions(args.npz)

    if args.out is None:
      out_dir = os.path.dirname(args.npz)
      out_html = os.path.join(out_dir, "full_transmission.html")
    else:
      out_html = args.out
      out_dir = os.path.dirname(out_html) or "."

    title = f"Full Transmission Spectrum | {os.path.basename(args.npz)}"
    html = build_html(
      energy_kev,
      full_transmission,
      title=title,
      window_kev=float(args.window_kev),
      step_fraction=float(args.step_fraction),
    )

    write_html(out_html, html)

    ion_files = []
    for ion_tag, ion_transmission in sorted(ion_transmissions.items()):
      ion_name = ion_display_name(ion_tag)
      ion_title = f"{ion_name} Transmission Spectrum | {os.path.basename(args.npz)}"
      ion_html = build_html(
        energy_kev,
        ion_transmission,
        title=ion_title,
        window_kev=float(args.window_kev),
        step_fraction=float(args.step_fraction),
      )
      ion_out = os.path.join(out_dir, ion_filename(ion_tag))
      write_html(ion_out, ion_html)
      ion_files.append(ion_out)

    print(f"Saved browser app: {out_html}")
    for ion_out in ion_files:
      print(f"Saved ion browser app: {ion_out}")
    print("Open this file in your browser.")


if __name__ == "__main__":
    main()
