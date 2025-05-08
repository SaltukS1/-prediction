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
                updatePredictions(data.predictions, homeTeam, awayTeam);
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
    
    function updatePredictions(predictions, homeTeam, awayTeam) {
        // Takım isimlerini güncelle
        document.getElementById('homeTeamName').textContent = homeTeam;
        document.getElementById('awayTeamName').textContent = awayTeam;
        
        // Maç sonucu tahminlerini güncelle
        const matchResult = predictions.match_result;
        updateProbability('winProb', 'winProbBar', matchResult.win_probability);
        updateProbability('drawProb', 'drawProbBar', matchResult.draw_probability);
        updateProbability('lossProb', 'lossProbBar', matchResult.loss_probability);
        
        // Skor tahminini güncelle
        const scorePrediction = document.getElementById('scorePrediction');
        scorePrediction.textContent = `${predictions.score_prediction.home} - ${predictions.score_prediction.away}`;
        
        // Toplam gol tahminini güncelle
        document.getElementById('expectedGoals').textContent = predictions.total_goals.expected;
        document.getElementById('goalRange').textContent = predictions.total_goals.range;
        
        // Alt/Üst gol tahminlerini güncelle
        updateOverUnderProbability('1_5', predictions.over_under_goals['1.5']);
        updateOverUnderProbability('2_5', predictions.over_under_goals['2.5']);
        updateOverUnderProbability('3_5', predictions.over_under_goals['3.5']);
        updateOverUnderProbability('4_5', predictions.over_under_goals['4.5']);
        
        // Karşılıklı gol tahminlerini güncelle
        document.getElementById('bttsYes').textContent = `${(predictions.btts_prediction.yes * 100).toFixed(1)}%`;
        document.getElementById('bttsNo').textContent = `${(predictions.btts_prediction.no * 100).toFixed(1)}%`;
        
        // Korner tahminlerini güncelle
        document.getElementById('totalCorners').textContent = predictions.corner_prediction.total_corners;
        
        // Korner alt/üst tahminlerini güncelle
        updateCornerProbability('3_5', predictions.corner_prediction.corner_ranges['3.5']);
        updateCornerProbability('4_5', predictions.corner_prediction.corner_ranges['4.5']);
        updateCornerProbability('5_5', predictions.corner_prediction.corner_ranges['5.5']);
        updateCornerProbability('8_5', predictions.corner_prediction.corner_ranges['8.5']);
        
        // İlk yarı tahminlerini güncelle
        const firstHalf = predictions.half_predictions.first_half;
        document.getElementById('firstHalfHomeWin').textContent = `${(firstHalf.home_win * 100).toFixed(1)}%`;
        document.getElementById('firstHalfDraw').textContent = `${(firstHalf.draw * 100).toFixed(1)}%`;
        document.getElementById('firstHalfAwayWin').textContent = `${(firstHalf.away_win * 100).toFixed(1)}%`;
        document.getElementById('firstHalfGoals').textContent = firstHalf.goals;
        
        // İkinci yarı tahminlerini güncelle
        const secondHalf = predictions.half_predictions.second_half;
        document.getElementById('secondHalfHomeWin').textContent = `${(secondHalf.home_win * 100).toFixed(1)}%`;
        document.getElementById('secondHalfDraw').textContent = `${(secondHalf.draw * 100).toFixed(1)}%`;
        document.getElementById('secondHalfAwayWin').textContent = `${(secondHalf.away_win * 100).toFixed(1)}%`;
        document.getElementById('secondHalfGoals').textContent = secondHalf.goals;
        
        // Gol atabilecek oyuncuları güncelle
        updateGoalscorers('homeTeamScorers', predictions.goalscorer_predictions.home_team);
        updateGoalscorers('awayTeamScorers', predictions.goalscorer_predictions.away_team);
    }
    
    function updateProbability(labelId, barId, probability) {
        const percentage = (probability * 100).toFixed(1);
        const label = document.getElementById(labelId);
        const bar = document.getElementById(barId);
        
        label.textContent = `${percentage}%`;
        bar.style.width = `${percentage}%`;
        bar.setAttribute('aria-valuenow', percentage);
    }
    
    function updateOverUnderProbability(threshold, probabilities) {
        const overPercentage = (probabilities.over * 100).toFixed(1);
        const underPercentage = (probabilities.under * 100).toFixed(1);
        
        document.getElementById(`over${threshold}`).textContent = `${overPercentage}%`;
        document.getElementById(`under${threshold}`).textContent = `${underPercentage}%`;
    }
    
    function updateCornerProbability(threshold, probabilities) {
        const overPercentage = (probabilities.over * 100).toFixed(1);
        const underPercentage = (probabilities.under * 100).toFixed(1);
        
        document.getElementById(`cornerOver${threshold}`).textContent = `${overPercentage}%`;
        document.getElementById(`cornerUnder${threshold}`).textContent = `${underPercentage}%`;
    }
    
    function updateGoalscorers(containerId, players) {
        const container = document.getElementById(containerId);
        container.innerHTML = ''; // Temizle
        
        // Gol olasılığına göre oyuncuları sırala
        const sortedPlayers = [...players].sort((a, b) => b.scoring_prob - a.scoring_prob);
        
        sortedPlayers.forEach(player => {
            const percentage = (player.scoring_prob * 100).toFixed(1);
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            // Oyuncu adı ve mevkisi
            const playerInfo = document.createElement('div');
            playerInfo.innerHTML = `
                <span class="fw-bold">${player.name}</span>
                <span class="badge bg-secondary ms-2">${player.position}</span>
            `;
            
            // Gol olasılığı
            const probBadge = document.createElement('span');
            
            // Olasılığa göre renk belirle
            let badgeClass = 'bg-info';
            if (percentage > 50) {
                badgeClass = 'bg-danger';
            } else if (percentage > 30) {
                badgeClass = 'bg-warning text-dark';
            }
            
            probBadge.className = `badge ${badgeClass}`;
            probBadge.textContent = `${percentage}%`;
            
            listItem.appendChild(playerInfo);
            listItem.appendChild(probBadge);
            container.appendChild(listItem);
        });
    }
});

// Takım isimlerini otomatik tamamlama için örnek veri
const teams = [
    'Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor',
    'Başakşehir', 'Alanyaspor', 'Sivasspor', 'Adana Demirspor',
    'Konyaspor', 'Antalyaspor', 'Kasımpaşa', 'Kayserispor',
    'Gaziantep FK', 'Hatayspor', 'Giresunspor', 'Ümraniyespor',
    'Barcelona', 'Real Madrid', 'Bayern Munich', 'Manchester City',
    'Liverpool', 'PSG', 'Inter Milan', 'Juventus', 'Arsenal',
    'Chelsea', 'Atletico Madrid', 'Borussia Dortmund', 'AC Milan'
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
document.addEventListener('DOMContentLoaded', function() {
    setupAutocomplete(document.getElementById('homeTeam'));
    setupAutocomplete(document.getElementById('awayTeam'));
}); 