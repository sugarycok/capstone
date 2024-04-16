from flask import Flask, render_template, request, jsonify
import FinanceDataReader as fdr

app = Flask(__name__, template_folder='templates')

# CORS 헤더 추가
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # 모든 도메인에서 요청 허용
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'  # GET과 POST 요청 허용
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Content-Type 헤더 허용
    return response

# 함수: 주식 종목 코드 가져오기
def fetch_stock_codes(company_name):
    # 뉴욕거래소와 나스닥에 상장된 기업 리스트 가져오기
    nyse_list = fdr.StockListing('NYSE')
    nasdaq_list = fdr.StockListing('NASDAQ')

    # 한국거래소에 상장된 기업 리스트 가져오기
    krx_list = fdr.StockListing('KRX')

    # 입력한 회사명으로 검색하여 종목 코드 가져오기
    matched_nyse = nyse_list[nyse_list['Name'].str.contains(company_name, case=False)]
    matched_nasdaq = nasdaq_list[nasdaq_list['Name'].str.contains(company_name, case=False)]
    matched_krx = krx_list[krx_list['Name'].str.contains(company_name, case=False)]

    stock_codes = []

    if not matched_nyse.empty:
        stock_codes.extend(matched_nyse['Symbol'].tolist())
    if not matched_nasdaq.empty:
        stock_codes.extend(matched_nasdaq['Symbol'].tolist())
    if not matched_krx.empty:
        stock_codes.extend(matched_krx['Code'].tolist())

    return stock_codes, company_name

# 라우트: 페이지에 대한 HTML 파일 렌더링
@app.route('/')
@app.route('/page4')
def render_page():
    return render_template('page4.html')

# 라우트: 주식 종목 코드 가져오기
@app.route('/get_stock_code', methods=['POST'])
def get_stock_code_route():
    data = request.get_json()
    company_name = data.get('company_name', '')

    stock_codes, company_name = fetch_stock_codes(company_name)
    if stock_codes:
        return jsonify({'stock_codes': stock_codes, 'company_name': company_name})
    else:
        return jsonify({'error': '해당 회사명의 종목을 찾을 수 없습니다.'}), 404

# 라우트: Python 코드에서 받아온 응답 반환하기
@app.route('/python_response')
def python_response():
    # 여기에 Python 코드에서 받아온 응답을 반환하는 코드를 추가하세요.
    
    # 임시로 예제 응답을 넣어봅시다.
    response_text = "Python response: This is the response from Python."

    return response_text

if __name__ == '__main__':
    app.run(debug=True)
