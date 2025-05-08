document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const predictionsDiv = document.getElementById('predictions');
    const loadingSpinner = document.getElementById('loadingSpinner');

    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const homeTeam = document.getElementById('homeTeam').value;
        const awayTeam = document.getElementById('awayTeam').value;
        
        if (!homeTeam || !awayTeam) {
            alert('Lütfen her iki takımı da seçin');
            return;
        }
        
        // Yükleme animasyonunu göster
        predictionsDiv.style.display = 'none';
        loadingSpinner.style.display = 'block';
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    home_team: homeTeam,
                    away_team: awayTeam
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                updatePredictions(data.predictions);
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            alert('Tahmin yapılırken bir hata oluştu: ' + error.message);
        } finally {
            loadingSpinner.style.display = 'none';
            predictionsDiv.style.display = 'block';
        }
    });
    
    function updatePredictions(predictions) {
        // Maç sonucu tahminlerini güncelle
        updateProgressBar('winProb', predictions.win_probability * 100);
        updateProgressBar('drawProb', predictions.draw_probability * 100);
        updateProgressBar('lossProb', predictions.loss_probability * 100);
        
        // Skor tahminini güncelle
        const scorePrediction = document.getElementById('scorePrediction');
        scorePrediction.textContent = `${predictions.score_prediction.home} - ${predictions.score_prediction.away}`;
        
        // Gol tahminlerini güncelle
        document.getElementById('overProb').textContent = `${(predictions.over_under_2_5.over * 100).toFixed(1)}%`;
        document.getElementById('underProb').textContent = `${(predictions.over_under_2_5.under * 100).toFixed(1)}%`;
        
        // Korner tahminlerini güncelle
        document.getElementById('totalCorners').textContent = predictions.corner_prediction.total_corners;
        document.getElementById('cornersOver8').textContent = 
            `${(predictions.corner_prediction.over_8_5 * 100).toFixed(1)}%`;
    }
    
    function updateProgressBar(elementId, percentage) {
        const progressBar = document.getElementById(elementId);
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.textContent = `${percentage.toFixed(1)}%`;
    }
});

// Takım isimlerini otomatik tamamlama için örnek veri
const teams = [
    'Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor',
    'Başakşehir', 'Alanyaspor', 'Sivasspor', 'Adana Demirspor',
    'Konyaspor', 'Antalyaspor', 'Kasımpaşa', 'Kayserispor',
    'Gaziantep FK', 'Hatayspor', 'Giresunspor', 'Ümraniyespor'
];

// Takım ismi girişlerini otomatik tamamlama özelliği
function setupAutocomplete(inputElement) {
    let currentFocus;
    
    inputElement.addEventListener('input', function(e) {
        let val = this.value;
        closeAllLists();
        
        if (!val) return false;
        currentFocus = -1;
        
        const matchList = document.createElement('div');
        matchList.setAttribute('id', this.id + 'autocomplete-list');
        matchList.setAttribute('class', 'autocomplete-items');
        this.parentNode.appendChild(matchList);
        
        for (let team of teams) {
            if (team.toLowerCase().includes(val.toLowerCase())) {
                const matchItem = document.createElement('div');
                const matchIndex = team.toLowerCase().indexOf(val.toLowerCase());
                
                matchItem.innerHTML = team.substr(0, matchIndex);
                matchItem.innerHTML += '<strong>' + team.substr(matchIndex, val.length) + '</strong>';
                matchItem.innerHTML += team.substr(matchIndex + val.length);
                
                matchItem.addEventListener('click', function(e) {
                    inputElement.value = team;
                    closeAllLists();
                });
                
                matchList.appendChild(matchItem);
            }
        }
    });
    
    function closeAllLists(elmnt) {
        const x = document.getElementsByClassName('autocomplete-items');
        for (let i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != inputElement) {
                x[i].parentNode.removeChild(x[i]);
            }
        }
    }
    
    document.addEventListener('click', function(e) {
        closeAllLists(e.target);
    });
}

// Otomatik tamamlama özelliğini her iki takım girişi için aktifleştir
setupAutocomplete(document.getElementById('homeTeam'));
setupAutocomplete(document.getElementById('awayTeam')); 