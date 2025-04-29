from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean

app = Flask(__name__)

def interpret_decision(f_value, critical_f, alpha):
    """Interpret the F-test decision with technical and plain language explanations."""
    if f_value > critical_f:
        return {
            'technical': "Reject H₀",
            'plain': "There is statistically significant evidence that the groups differ",
            'confidence': f"At {alpha*100}% significance, we conclude means are not all equal"
        }
    return {
        'technical': "Accept H₀",
        'plain': "Insufficient evidence to conclude groups differ",
        'confidence': f"At {alpha*100}% significance, we cannot reject equal means"
    }

"""@app.route('/', methods=['GET', 'POST'])
def index():
   
    # Initialize all template variables with safe defaults
    context = {
        'result': {},
        'plot_div': '',
        'error': None
    }
    """

"""@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'result': {},
        'plot_div': None,
        'error': None,
        'form_data': request.form if request.method == 'POST' else None
       

    }
    
    
    if request.method == 'POST':
        test_type = request.form.get('test_type', '')
        raw_data = request.form.get('raw_data', '')
        df = pd.read_csv(file)
        data = df.to_numpy()

        try:
            # Process input data with missing value handling
            data = []
            for row in raw_data.strip().split('\n'):
                row_data = []
                for num in row.split(','):
                    stripped_num = num.strip()
                    if stripped_num:
                        try:
                            row_data.append(float(stripped_num))
                        except ValueError:
                            row_data.append(np.nan)
                if row_data:
                    data.append(row_data)
            
            if len(data) < 2:
                context['error'] = "At least two groups required for ANOVA"
                return render_template('index.html', **context)
            
            if test_type == 'one_way':
                context['result']= one_way_classification(data) 
                context['plot_div'] = create_box_plot(data) or ''
            elif test_type == 'two_way':
                context['result'] = two_way_classification(data) or {}
                context['plot_div'] = create_heatmap(data) or ''
            if not context['error']:
                # Add anchor to redirect URL
                return render_template('index.html', **context, _anchor='result-scroll')
                
        except Exception as e:
            context['error'] = f"Error processing data: {str(e)}"
            app.logger.error(f"Error in index route: {str(e)}")
    
    return render_template('index.html', **context)"""
"""@app.route('/oneway', methods=['POST'])
def handle_oneway():
    raw_data = request.form['data']
    axis = int(request.form.get('axis', 0))
    # Convert raw_data to list of lists...
    result = one_way_classification(raw_data, axis=axis)
    return render_template('result.html', result=result)"""
""""@app.route('/oneway', methods=['POST'])
def oneway():
    try:
        input_data = request.get_json()
        data = input_data['data']
        alpha = float(input_data.get('alpha', 0.05))
        axis = int(input_data.get('axis', 0))  # Make sure this is 0 if columns are groups
        result = one_way_classification(data, alpha=alpha, axis=axis)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})"""

'''def one_way_classification(data, alpha=0.05, axis=0):
    try:
        data = np.array(data, dtype=np.float64)

        grouped_data = []
        if axis == 1:
            for row in data:
                cleaned = row[~np.isnan(row)]
                if len(cleaned) > 0:
                    grouped_data.append(cleaned.tolist())
        else:
            for col in data.T:
                cleaned = col[~np.isnan(col)]
                if len(cleaned) > 0:
                    grouped_data.append(cleaned.tolist())

        all_data = [x for group in grouped_data for x in group]
        N = sum(len(group) for group in grouped_data)
        T = sum(all_data)
        T_square_by_N = (T ** 2) / N
        sum_of_squares = sum(x ** 2 for x in all_data)
        TSS = sum_of_squares - T_square_by_N
        SSC = sum((sum(group) ** 2) / len(group) for group in grouped_data) - T_square_by_N
        SSE = TSS - SSC

        df_treatment = len(grouped_data) - 1
        df_error = N - len(grouped_data)
        MSC = SSC / df_treatment
        MSE = SSE / df_error

        F = MSC / MSE
        CRITICAL_F = stats.f.ppf(1 - alpha, df_treatment, df_error)
        decision = interpret_decision(F, CRITICAL_F, alpha)

        group_stats = [{
            'group': f"Group {i+1}",
            'count': len(group),
            'mean': round(mean(group), 4) if group else 'N/A',
            'variance': round(np.var(group, ddof=1), 4) if len(group) > 1 else 'N/A'
        } for i, group in enumerate(grouped_data)]

        return {
            'Test Type': 'One-Way ANOVA',
            'Group Statistics': group_stats,
            'N (Total Observations)': N,
            'T (Total Sum)': T,
            'T²/N': round(T_square_by_N, 4),
            'TSS': round(TSS, 4),
            'SSC': round(SSC, 4),
            'SSE': round(SSE, 4),
            'Degrees of Freedom (Column)': df_treatment,
            'Degrees of Freedom (Error)': df_error,
            'MSC': round(MSC, 4),
            'MSE': round(MSE, 4),
            'F-Statistic (Column)': round(F, 4),
            'Critical F-Value (Column)': round(CRITICAL_F, 4),
            'Column Technical Decision': decision['technical'],
            'Column Plain Language Conclusion': decision['plain'],
            'Column Confidence Statement': decision['confidence'],
            'Alpha Level': alpha
        }

    except Exception as e:
        return {'error': str(e)}'''
"""data = [
   [1,-2,-14,-9],
   [1,4,-5,-8],
   [5,4,0,-7],
   [8,10,2,-3],
   [10,15,4,0],
   [12, None,6,8],
   [20,None,14,None],
   [None,None,22,None]
]
result = one_way_classification(data, axis=0)

print(result)"""
@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'result': {},
        'plot_div': None,
        'error': None,
        'form_data': request.form if request.method == 'POST' else None
    }

    if request.method == 'POST':
        test_type = request.form.get('test_type', '')
        alpha = float(request.form.get('alpha', 0.05))
        axis = int(request.form.get('axis', 0))
        raw_data = request.form.get('raw_data', '').strip()
        uploaded_file = request.files.get('file')

        try:
            # 1. Load data from uploaded CSV file if available
            if uploaded_file and uploaded_file.filename != '':
                df = pd.read_csv(uploaded_file)
                data = df.to_numpy()
            else:
                # 2. Else, parse from raw textarea input
                data = []
                for row in raw_data.split('\n'):
                    row_data = []
                    for num in row.split(','):
                        stripped = num.strip()
                        if stripped:
                            try:
                                row_data.append(float(stripped))
                            except ValueError:
                                row_data.append(np.nan)
                    if row_data:
                        data.append(row_data)
                data = np.array(data, dtype=np.float64)

            # 3. Minimum check
            if len(data) < 2:
                context['error'] = "At least two groups required for ANOVA"
                return render_template('index.html', **context)

            # 4. Test selection
            if test_type == 'one_way':
                context['result'] = one_way_classification(data, alpha=alpha, axis=axis)
                context['plot_div'] = create_box_plot(data) or ''
            elif test_type == 'two_way':
                context['result'] = two_way_classification(data) or {}
                context['plot_div'] = create_heatmap(data) or ''
            else:
                context['error'] = "Invalid test type selected"

            return render_template('index.html', **context, _anchor='result-scroll')

        except Exception as e:
            context['error'] = f"Error processing data: {str(e)}"
            app.logger.error(f"Error in index route: {str(e)}")

    return render_template('index.html', **context)


def one_way_classification(data, alpha=0.05,axis=0):

    
 
    try:
        # Clean data by removing NaN values
        cleaned_data = [
            [x for x in row if not np.isnan(x)] 
            for row in data 
            if any(not np.isnan(x) for x in row)
        ]
        
        if len(cleaned_data) < 2:
            return {'error': 'Insufficient valid data after cleaning'}
        
        # Organize data into groups
        grouped_data = [[] for _ in range(max(len(row) for row in cleaned_data))]
        for row in cleaned_data:
            for j, val in enumerate(row):
                if j < len(grouped_data):
                    grouped_data[j].append(val)
        
        grouped_data = [group for group in grouped_data if group]
        all_data = [val for group in grouped_data for val in group]
    
        # Calculate ANOVA components
        #N = len(all_data)
        N = sum(len(group) for group in grouped_data)

        T = sum(all_data)
        
        T_square_by_N = (T ** 2) / N
        sum_of_squares = sum(x ** 2 for x in all_data)
        TSS = sum_of_squares - T_square_by_N
        SSC = sum((sum(group) ** 2) / len(group) for group in grouped_data) - T_square_by_N
        SSE = TSS - SSC
        
        df_treatment = len(grouped_data) - 1
        df_error = N - len(grouped_data)
        MSC = SSC / df_treatment
        MSE = SSE / df_error
        
        F = MSC / MSE if MSC >= MSE else MSE / MSC
        df_num = df_treatment if MSC >= MSE else df_error
        df_den = df_error if MSC >= MSE else df_treatment
        
        CRITICAL_F = stats.f.ppf(1 - alpha, df_num, df_den)
        decision = interpret_decision(F, CRITICAL_F, alpha)

        # Prepare group statistics
        group_stats = [{
            'group': f"Group {i+1}",
            'count': len(group),
            'mean': round(mean(group), 4) if group else 'N/A',
            'variance': round(np.var(group, ddof=1), 4) if len(group) > 1 else 'N/A'
        } for i, group in enumerate(grouped_data)]

        return {
            'Test Type': 'One-Way ANOVA',
            'Group Statistics': group_stats,
            'N (Total Observations)': N,
            'T (Total Sum)': T,
            'T²/N': round(T_square_by_N, 4),
            'TSS': round(TSS, 4),
            'SSC': round(SSC, 4),
            'SSE': round(SSE, 4),
            'Degrees of Freedom (Column)': df_treatment,
            'Degrees of Freedom (Error)': df_error,
            'MSC': round(MSC, 4),
            'MSE': round(MSE, 4),
            'F-Statistic (Column)': round(F, 4),
            'Critical F-Value (Column)': round(CRITICAL_F, 4),
            'Column Technical Decision': decision['technical'],
            'Column Plain Language Conclusion': decision['plain'],
            
            
        }
    except Exception as e:
        app.logger.error(f"Error in one_way_classification: {str(e)}")
        return {'error': str(e)}

def two_way_classification(data, alpha=0.05):
    # Handle missing values by removing them
    cleaned_data = []
    for row in data:
        cleaned_row = [x if not np.isnan(x) else None for x in row]
        cleaned_data.append(cleaned_row)
    
    # Flatten data and ignore missing values
    values = []
    for i in range(len(cleaned_data)):
        for j in range(len(cleaned_data[i])):
            val = cleaned_data[i][j]
            if val is not None:
                values.append((i, j, val))

    N = len(values)
    if N < 2:
        return {'error': 'Not enough valid data after removing missing values'}

    all_values = [val for _, _, val in values]
    T = sum(all_values)
    T_square_by_N = (T ** 2) / N
    grand_mean = np.mean(all_values)

    # Row and column statistics
    row_stats = {}
    col_stats = {}
    row_totals = {}
    col_totals = {}
    row_counts = {}
    col_counts = {}

    for i, j, val in values:
        # Row calculations
        row_totals[i] = row_totals.get(i, 0) + val
        row_counts[i] = row_counts.get(i, 0) + 1
        
        # Column calculations
        col_totals[j] = col_totals.get(j, 0) + val
        col_counts[j] = col_counts.get(j, 0) + 1

    # Calculate row means for reporting
    for i in row_totals:
        row_stats[f"Row {i+1}"] = {
            'count': row_counts[i],
            'mean': round(row_totals[i] / row_counts[i], 4)
        }

    # Calculate column means for reporting
    for j in col_totals:
        col_stats[f"Column {j+1}"] = {
            'count': col_counts[j],
            'mean': round(col_totals[j] / col_counts[j], 4)
        }

    ss_total = sum((val - grand_mean) ** 2 for val in all_values)

    ss_row = sum((row_totals[i] ** 2) / row_counts[i] for i in row_totals) - (sum(all_values) ** 2) / N
    ss_col = sum((col_totals[j] ** 2) / col_counts[j] for j in col_totals) - (sum(all_values) ** 2) / N
    ss_error = ss_total - ss_row - ss_col

    df_row = len(row_totals) - 1
    df_col = len(col_totals) - 1
    df_error = N - len(row_totals) - len(col_totals) + 1

    ms_row = ss_row / df_row
    ms_col = ss_col / df_col
    ms_error = ss_error / df_error

    # Row factor test
    if ms_row >= ms_error:
        f_row = ms_row / ms_error
        df_row_num = df_row
        df_row_den = df_error
    else:
        f_row = ms_error / ms_row
        df_row_num = df_error
        df_row_den = df_row

    critical_f_row = stats.f.ppf(1 - alpha, df_row_num, df_row_den)
    row_decision = interpret_decision(f_row, critical_f_row, alpha)

    # Column factor test
    if ms_col >= ms_error:
        f_col = ms_col / ms_error
        df_col_num = df_col
        df_col_den = df_error
    else:
        f_col = ms_error / ms_col
        df_col_num = df_error
        df_col_den = df_col

    critical_f_col = stats.f.ppf(1 - alpha, df_col_num, df_col_den)
    col_decision = interpret_decision(f_col, critical_f_col, alpha)

    return {
        'Test Type': 'Two-Way ANOVA (Unbalanced)',
        'Row Statistics': row_stats,
        'Column Statistics': col_stats,
        'N (Total Observations)': N,
        'T (Total Sum)': T,
        'T²/N': round(T_square_by_N, 4),
        'TSS (Total Sum of Squares)': round(ss_total, 4),
        'SSR': round(ss_row, 4),
        'SSC': round(ss_col, 4),
        'SSE': round(ss_error, 4),
        'Degrees of Freedom (Row)': df_row,
        'Degrees of Freedom (Column)': df_col,
        'Degrees of Freedom (Error)': df_error,
        'MSC':round(ms_col,4),
        'MSR':round(ms_row,4),
        'MSE':round(ms_error,4),
        'F-Statistic (Row)': round(f_row, 4),
        'Critical F-Value (Row)': round(critical_f_row, 4),
        'Row Technical Decision': row_decision['technical'],
        'Row Plain Language Conclusion': row_decision['plain'],
        
        'F-Statistic (Column)': round(f_col, 4),
        'Critical F-Value (Column)': round(critical_f_col, 4),
        'Column Technical Decision': col_decision['technical'],
        'Column Plain Language Conclusion': col_decision['plain'],
        
        
    }

def create_box_plot(data):
    # Handle missing values
    cleaned_data = []
    for row in data:
        cleaned_row = [x for x in row if not np.isnan(x)]
        if cleaned_row:
            cleaned_data.append(cleaned_row)
    
    fig = go.Figure()
    for i, group in enumerate(cleaned_data):
        fig.add_trace(go.Box(
            y=group,
            name=f'Group {i+1}',
            boxmean='sd',
            marker_color='#4361ee',
            line_color='#3f37c9'
        ))
    fig.update_layout(
        title="Box Plot of Group Distributions",
        xaxis_title="Groups",
        yaxis_title="Values",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#212529')
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_heatmap(data):
    # Create a matrix with NaN for missing values
    max_cols = max(len(row) for row in data)
    matrix_data = np.full((len(data), max_cols), np.nan)
    
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if not np.isnan(val):
                matrix_data[i,j] = val
    
    fig = px.imshow(
        matrix_data, 
        labels=dict(x="Columns", y="Rows", color="Values"),
        title="Heatmap of Two-Way Data (White = Missing Values)",
        color_continuous_scale='blues',
        text_auto=".2f"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#212529')
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')



if __name__ == "__main__":
    app.run(debug=True)


