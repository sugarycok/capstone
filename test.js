// test.js
const express = require('express');
const app = express();
const path = require('path');
const PORT = 3000; // 사용할 포트 번호

// 정적 파일들을 제공하기 위한 미들웨어
app.use(express.static(path.join(__dirname, 'static')));

// 메인 페이지
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

// 각각의 메뉴 항목에 대한 라우팅
app.get('/page1', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'page1.html'));
});

app.get('/page2', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'page2.html'));
});

app.get('/page3', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'page3.html'));
});

app.get('/page4', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'page4.html'));
});

// 서버를 시작합니다.
app.listen(PORT, () => {
    console.log(`서버가 http://localhost:${PORT} 에서 실행 중입니다!`);
});
