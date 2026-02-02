import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
from io import BytesIO
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Quadrant Generator", layout="centered")

st.title("🎯 Strategic Quadrant Map Generator")
st.markdown("""
**Instructions:**
1. Upload an Excel or CSV file.
2. Ensure it has columns for **Label**, **Performance (X)**, and **Importance (Y)**.
3. The app will automatically calculate averages, draw the red dashed lines, and generate your files.
""")
st.divider()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Data File (CSV or XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --- COLUMN SELECTION ---
    st.write("### ⚙️ Column Mapping")
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    label_col = c1.selectbox("Select Label Column", cols, index=0)
    x_col = c2.selectbox("Select Performance Column (X-Axis)", cols, index=1 if len(cols) > 1 else 0)
    y_col = c3.selectbox("Select Importance Column (Y-Axis)", cols, index=2 if len(cols) > 2 else 0)

    st.divider()
    if st.button("🎨 Generate Quadrant Map", type="primary", use_container_width=True):
        # --- CALCULATIONS ---
        avg_x = df[x_col].mean()
        avg_y = df[y_col].mean()

        # Axis Limits (Min - 15%, Max + 15%)
        x_range = df[x_col].max() - df[x_col].min()
        y_range = df[y_col].max() - df[y_col].min()

        x_min_raw = df[x_col].min() - (x_range * 0.15)
        x_max_raw = df[x_col].max() + (x_range * 0.15)
        y_min_raw = df[y_col].min() - (y_range * 0.15)
        y_max_raw = df[y_col].max() + (y_range * 0.15)

        # Round bounds to clean values for Excel compatibility
        # Round min DOWN to nearest 0.5%, max UP to nearest 0.5%
        x_min = np.floor(x_min_raw * 200) / 200  # Round down to nearest 0.005 (0.5%)
        x_max = np.ceil(x_max_raw * 200) / 200   # Round up to nearest 0.005 (0.5%)
        y_min = np.floor(y_min_raw * 200) / 200  # Round down to nearest 0.005 (0.5%)
        y_max = np.ceil(y_max_raw * 200) / 200   # Round up to nearest 0.005 (0.5%)

        # --- 1. GENERATE PNG IMAGE ---
        fig, ax = plt.subplots(figsize=(14, 11))

        # Heatmap Shading (Professional Standard)
        ax.fill_between([avg_x, x_max], avg_y, y_max, color='#e6f4ea', alpha=0.4) # Top-Right (Green)
        ax.fill_between([x_min, avg_x], avg_y, y_max, color='#fff4e5', alpha=0.4) # Top-Left (Orange)
        ax.fill_between([x_min, avg_x], y_min, avg_y, color='#f1f3f4', alpha=0.4) # Bottom-Left (Gray)
        ax.fill_between([avg_x, x_max], y_min, avg_y, color='#e8f0fe', alpha=0.4) # Bottom-Right (Blue)

        # Data Points
        ax.scatter(df[x_col], df[y_col], color='#1a73e8', s=120, edgecolors='white', linewidth=1.5, zorder=5)

        # MANDATORY: Red Dashed Axes
        ax.axvline(x=avg_x, color='#d93025', linestyle='--', linewidth=2, zorder=3)
        ax.axhline(y=avg_y, color='#d93025', linestyle='--', linewidth=2, zorder=3)

        # Labels with Callouts
        # Simple offset logic to prevent total overlap
        offsets = [(15, 15), (-15, 20), (20, -15), (-20, -15), (25, 5), (-25, 5), (5, 25), (5, -25)]
        for i, txt in enumerate(df[label_col]):
            off = offsets[i % len(offsets)]
            ax.annotate(txt, (df[x_col].iloc[i], df[y_col].iloc[i]),
                        xytext=off, textcoords='offset points',
                        fontsize=9, fontweight='bold', color='#3c4043',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#dadce0', alpha=0.9),
                        arrowprops=dict(arrowstyle='-', color='#9aa0a6', lw=0.8),
                        zorder=10)

        # Formatting
        ax.set_title(f'Strategic Analysis (X Avg: {avg_x:.1%}, Y Avg: {avg_y:.1%})', fontsize=20, fontweight='bold', pad=30)
        ax.set_xlabel(x_col, fontsize=14, labelpad=10)
        ax.set_ylabel(y_col, fontsize=14, labelpad=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Force % Formatting
        vals_x = ax.get_xticks()
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals_x])
        vals_y = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in vals_y])

        # Corner Labels
        ax.text(0.98, 0.98, 'KEY STRENGTHS', transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold', color='#137333')
        ax.text(0.02, 0.98, 'PRIORITY IMPROVEMENTS', transform=ax.transAxes, ha='left', va='top', fontsize=12, fontweight='bold', color='#b06000')
        ax.text(0.02, 0.02, 'LOW PRIORITY', transform=ax.transAxes, ha='left', va='bottom', fontsize=12, fontweight='bold', color='#5f6368')
        ax.text(0.98, 0.02, 'SECONDARY ASSETS', transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold', color='#174ea6')

        ax.grid(visible=False)
        plt.tight_layout()

        # Display Image on Web App
        st.pyplot(fig)

        # Save Image to Buffer for Download
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=200)
        img_buffer.seek(0)

        # --- 2. GENERATE EXCEL FILE ---
        excel_buffer = BytesIO()
        workbook = xlsxwriter.Workbook(excel_buffer, {'in_memory': True})
        ws = workbook.add_worksheet('Analysis')

        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#f8f9fa', 'border': 1, 'align': 'center'})
        pct_fmt = workbook.add_format({'num_format': '0.0%', 'border': 1, 'align': 'center'})
        txt_fmt = workbook.add_format({'border': 1})

        ws.write(0, 0, 'Statement', header_fmt)
        ws.write(0, 1, x_col, header_fmt)
        ws.write(0, 2, y_col, header_fmt)

        for i in range(len(df)):
            ws.write(i + 1, 0, str(df[label_col].iloc[i]), txt_fmt)
            ws.write(i + 1, 1, df[x_col].iloc[i], pct_fmt)
            ws.write(i + 1, 2, df[y_col].iloc[i], pct_fmt)

        # Reference values for verification
        row_count = len(df)
        ws.write(row_count + 3, 0, 'Calculated Averages:', header_fmt)
        ws.write(row_count + 3, 1, f'X Avg: {avg_x:.2%}', txt_fmt)
        ws.write(row_count + 3, 2, f'Y Avg: {avg_y:.2%}', txt_fmt)

        ws.write(row_count + 4, 0, 'Axis Ranges:', header_fmt)
        ws.write(row_count + 4, 1, f'X: {x_min:.2%} to {x_max:.2%}', txt_fmt)
        ws.write(row_count + 4, 2, f'Y: {y_min:.2%} to {y_max:.2%}', txt_fmt)

        # Hidden Data for Red Lines - use NO formatting (raw values only)
        # This ensures Excel plots them exactly as-is without any conversion

        # Vertical line: X=avg_x from y_min to y_max
        ws.write(row_count + 7, 5, avg_x)  # X coord point 1
        ws.write(row_count + 8, 5, avg_x)  # X coord point 2
        ws.write(row_count + 7, 6, y_min)  # Y coord point 1
        ws.write(row_count + 8, 6, y_max)  # Y coord point 2

        # Horizontal line: Y=avg_y from x_min to x_max
        ws.write(row_count + 7, 8, x_min)  # X coord point 1
        ws.write(row_count + 8, 8, x_max)  # X coord point 2
        ws.write(row_count + 7, 9, avg_y)  # Y coord point 1
        ws.write(row_count + 8, 9, avg_y)  # Y coord point 2

        # Create Chart (scatter plot with markers only, no connecting lines)
        chart = workbook.add_chart({'type': 'scatter'})
        
        # Custom Labels with Leader Lines
        custom_labels = []
        for i in range(row_count):
            custom_labels.append({
                'value': f'=Analysis!$A${i+2}', 
                'font': {'size': 9, 'bold': True}, 
                'border': {'color': '#bfbfbf'}, 
                'fill': {'color': 'white'}
            })

        chart.add_series({
            'name': 'Data',
            'categories': ['Analysis', 1, 1, row_count, 1],
            'values':     ['Analysis', 1, 2, row_count, 2],
            'data_labels': {'custom': custom_labels, 'leader_lines': True},
            'marker': {'type': 'circle', 'size': 10, 'fill': {'color': '#1a73e8'}, 'border': {'color': 'white'}}
        })

        # Red Axes Lines
        # Vertical line at X average
        chart.add_series({
            'categories': ['Analysis', row_count+7, 5, row_count+8, 5],
            'values': ['Analysis', row_count+7, 6, row_count+8, 6],
            'line': {'color': 'red', 'dash_type': 'dash', 'width': 2},
            'marker': {'type': 'none'}
        })
        # Horizontal line at Y average
        chart.add_series({
            'categories': ['Analysis', row_count+7, 8, row_count+8, 8],
            'values': ['Analysis', row_count+7, 9, row_count+8, 9],
            'line': {'color': 'red', 'dash_type': 'dash', 'width': 2},
            'marker': {'type': 'none'}
        })

        # Set chart title
        chart.set_title({'name': f'Strategic Analysis (X={avg_x:.1%}, Y={avg_y:.1%})'})

        # Write axis bounds for reference
        ws.write(row_count + 10, 0, 'Expected X Range:', header_fmt)
        ws.write(row_count + 10, 1, f'{x_min:.2%} to {x_max:.2%}', txt_fmt)
        ws.write(row_count + 11, 0, 'Expected Y Range:', header_fmt)
        ws.write(row_count + 11, 1, f'{y_min:.2%} to {y_max:.2%}', txt_fmt)

        # Set axis bounds with padding (same as PNG chart)
        # Use same bounds as the hidden data to ensure consistency
        chart.set_x_axis({
            'name': x_col,
            'min': x_min,
            'max': x_max,
            'num_format': '0%',
            'label_position': 'low'
        })
        chart.set_y_axis({
            'name': y_col,
            'min': y_min,
            'max': y_max,
            'num_format': '0%',
            'label_position': 'low'
        })
        chart.set_size({'width': 1100, 'height': 850})
        chart.set_legend({'none': True})

        ws.insert_chart('E2', chart)
        workbook.close()

        st.success("✅ Analysis Complete!")
        st.info(f"📈 Averages: X = {avg_x:.1%} | Y = {avg_y:.1%}")

        # DOWNLOAD BUTTONS
        st.write("### 💾 Download Your Files")
        c_d1, c_d2 = st.columns(2)
        c_d1.download_button(
            "📸 Download PNG Image",
            img_buffer,
            "Quadrant_Map.png",
            "image/png",
            use_container_width=True
        )
        c_d2.download_button(
            "📊 Download Excel File",
            excel_buffer,
            "Quadrant_Analysis.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )