document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    const predictionsDiv = document.getElementById('predictions');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const predictionHistory = document.getElementById('predictionHistory');
    const clearHistoryBtn = document.getElementById('clearHistory');
    
    // Favori takımlar sistemi
    const favoriteTeams = JSON.parse(localStorage.getItem('favoriteTeams')) || [];
    setupFavorites();
    
    // Tahmin geçmişi
    const matchHistory = JSON.parse(localStorage.getItem('matchHistory')) || [];
    setupHistory();

    // Canlı maç simülasyonu değişkenleri
    let simulationInterval;
    let matchMinute = 0;
    let homeScore = 0;
    let awayScore = 0;
    let homePossession = 50;
    let lastPredictions = null;
    let currentHomeTeam = '';
    let currentAwayTeam = '';
    
    // Canlı maç simülasyonu butonları
    const startSimulationBtn = document.getElementById('startSimulation');
    const stopSimulationBtn = document.getElementById('stopSimulation');
    
    if (startSimulationBtn) {
        startSimulationBtn.addEventListener('click', startMatchSimulation);
    }
    
    if (stopSimulationBtn) {
        stopSimulationBtn.addEventListener('click', stopMatchSimulation);
    }

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
                lastPredictions = data.predictions;
                currentHomeTeam = homeTeam;
                currentAwayTeam = awayTeam;
                
                // Canlı maç bölümünü hazırla ve göster
                document.getElementById('liveMatchSection').style.display = 'block';
                document.getElementById('liveHomeTeam').textContent = homeTeam;
                document.getElementById('liveAwayTeam').textContent = awayTeam;
                document.getElementById('liveScore').textContent = '0 - 0';
                document.getElementById('matchTime').textContent = "0'";
                document.getElementById('matchEvents').innerHTML = '<li class="list-group-item p-2">Maç henüz başlamadı</li>';
                
                // Diğer güncellemeler...
                updatePredictions(data.predictions, homeTeam, awayTeam);
                
                // Tahmin geçmişine ekle
                addToHistory(homeTeam, awayTeam, data.predictions);
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
        
        // Takım karşılaştırma bölümünü göster ve güncelle
        updateTeamComparison(predictions.team_comparison || {
            home_team: {
                form: Math.round(predictions.match_result.win_probability * 100),
                season_points: Math.round(matchResult.win_probability * 100),
                goals_scored_avg: predictions.total_goals.home_expected || 1.5,
                goals_conceded_avg: 1.0,
                last_matches: generateMockMatches(matchResult.win_probability)
            },
            away_team: {
                form: Math.round(predictions.match_result.loss_probability * 100),
                season_points: Math.round(matchResult.loss_probability * 90),
                goals_scored_avg: predictions.total_goals.away_expected || 1.0,
                goals_conceded_avg: 1.2,
                last_matches: generateMockMatches(matchResult.loss_probability)
            }
        }, homeTeam, awayTeam);
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

    // Favorilere takım ekleme/çıkarma için fonksiyonlar
    function setupFavorites() {
        // Favori butonları ekle
        const homeTeamInput = document.getElementById('homeTeam');
        const awayTeamInput = document.getElementById('awayTeam');
        
        addFavoriteButton(homeTeamInput, 'homeTeamFav');
        addFavoriteButton(awayTeamInput, 'awayTeamFav');
        
        // Favori takımlar menüsünü oluştur
        createFavoritesList();
    }
    
    function addFavoriteButton(inputElement, buttonId) {
        const parentDiv = inputElement.parentNode;
        
        // Input grubunu oluştur
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        // Input elementini input grubuna taşı
        inputElement.parentNode.insertBefore(inputGroup, inputElement);
        inputGroup.appendChild(inputElement);
        
        // Favori butonu oluştur
        const favButton = document.createElement('button');
        favButton.className = 'btn btn-outline-secondary dropdown-toggle';
        favButton.type = 'button';
        favButton.id = buttonId;
        favButton.dataset.bsToggle = 'dropdown';
        favButton.innerHTML = '<i class="fas fa-star"></i>';
        
        // Dropdown menü oluştur
        const dropdownMenu = document.createElement('ul');
        dropdownMenu.className = 'dropdown-menu dropdown-menu-end';
        dropdownMenu.id = buttonId + 'Menu';
        
        // Favori ekle seçeneği
        const addFavItem = document.createElement('li');
        const addFavLink = document.createElement('a');
        addFavLink.className = 'dropdown-item';
        addFavLink.href = '#';
        addFavLink.textContent = 'Favorilere Ekle';
        addFavLink.onclick = function() {
            const teamName = inputElement.value;
            if (teamName && !favoriteTeams.includes(teamName)) {
                favoriteTeams.push(teamName);
                localStorage.setItem('favoriteTeams', JSON.stringify(favoriteTeams));
                createFavoritesList();
            }
            return false;
        };
        addFavItem.appendChild(addFavLink);
        dropdownMenu.appendChild(addFavItem);
        
        // Favori takımlar ayırıcı
        const divider = document.createElement('li');
        divider.innerHTML = '<hr class="dropdown-divider">';
        dropdownMenu.appendChild(divider);
        
        // Input grubuna buton ve menüyü ekle
        inputGroup.appendChild(favButton);
        inputGroup.appendChild(dropdownMenu);
    }
    
    function createFavoritesList() {
        // Her iki takım için de favori listelerini güncelle
        updateFavoriteMenu('homeTeamFavMenu');
        updateFavoriteMenu('awayTeamFavMenu');
    }
    
    function updateFavoriteMenu(menuId) {
        const menu = document.getElementById(menuId);
        if (!menu) return;
        
        // İlk iki öğeyi tut (Favorilere Ekle ve ayırıcı), geri kalan favorileri temizle
        while (menu.children.length > 2) {
            menu.removeChild(menu.lastChild);
        }
        
        // Favori takımlar yoksa mesaj göster
        if (favoriteTeams.length === 0) {
            const emptyItem = document.createElement('li');
            const emptyLink = document.createElement('a');
            emptyLink.className = 'dropdown-item disabled';
            emptyLink.href = '#';
            emptyLink.textContent = 'Favori takım yok';
            emptyItem.appendChild(emptyLink);
            menu.appendChild(emptyItem);
            return;
        }
        
        // Favori takımları ekle
        favoriteTeams.forEach(team => {
            const teamItem = document.createElement('li');
            const teamContainer = document.createElement('div');
            teamContainer.className = 'dropdown-item d-flex justify-content-between align-items-center';
            
            const teamLink = document.createElement('a');
            teamLink.href = '#';
            teamLink.textContent = team;
            teamLink.onclick = function() {
                const inputId = menuId === 'homeTeamFavMenu' ? 'homeTeam' : 'awayTeam';
                document.getElementById(inputId).value = team;
                return false;
            };
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'btn btn-sm btn-outline-danger ms-2';
            removeBtn.innerHTML = '<i class="fas fa-times"></i>';
            removeBtn.onclick = function() {
                const index = favoriteTeams.indexOf(team);
                if (index > -1) {
                    favoriteTeams.splice(index, 1);
                    localStorage.setItem('favoriteTeams', JSON.stringify(favoriteTeams));
                    createFavoritesList();
                }
                return false;
            };
            
            teamContainer.appendChild(teamLink);
            teamContainer.appendChild(removeBtn);
            teamItem.appendChild(teamContainer);
            menu.appendChild(teamItem);
        });
    }

    // Tahmin geçmişi fonksiyonları
    function setupHistory() {
        // Geçmiş tahminleri göster
        updateHistoryDisplay();
        
        // Temizleme butonunu ayarla
        clearHistoryBtn.addEventListener('click', function() {
            if (confirm('Tüm tahmin geçmişiniz silinecek. Emin misiniz?')) {
                matchHistory.length = 0;
                localStorage.setItem('matchHistory', JSON.stringify(matchHistory));
                updateHistoryDisplay();
            }
        });
    }
    
    function addToHistory(homeTeam, awayTeam, predictions) {
        const timestamp = new Date().toISOString();
        const scorePrediction = predictions.score_prediction;
        
        const historyItem = {
            timestamp: timestamp,
            homeTeam: homeTeam,
            awayTeam: awayTeam,
            score: `${scorePrediction.home}-${scorePrediction.away}`,
            winProb: (predictions.match_result.win_probability * 100).toFixed(1)
        };
        
        // En başa ekle ve maksimum 10 kayıt tut
        matchHistory.unshift(historyItem);
        if (matchHistory.length > 10) {
            matchHistory.pop();
        }
        
        localStorage.setItem('matchHistory', JSON.stringify(matchHistory));
        updateHistoryDisplay();
    }
    
    function updateHistoryDisplay() {
        predictionHistory.innerHTML = '';
        
        if (matchHistory.length === 0) {
            const emptyItem = document.createElement('li');
            emptyItem.className = 'list-group-item text-center text-muted';
            emptyItem.textContent = 'Henüz tahmin yapılmadı';
            predictionHistory.appendChild(emptyItem);
            return;
        }
        
        matchHistory.forEach(item => {
            const historyItem = document.createElement('li');
            historyItem.className = 'list-group-item';
            
            const date = new Date(item.timestamp);
            const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
            
            historyItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <small class="text-muted">${formattedDate}</small>
                        <div>${item.homeTeam} - ${item.awayTeam}</div>
                    </div>
                    <div class="text-end">
                        <div class="fw-bold">${item.score}</div>
                        <span class="badge bg-primary">${item.winProb}%</span>
                    </div>
                </div>
            `;
            
            historyItem.addEventListener('click', function() {
                document.getElementById('homeTeam').value = item.homeTeam;
                document.getElementById('awayTeam').value = item.awayTeam;
                predictionForm.dispatchEvent(new Event('submit'));
            });
            
            predictionHistory.appendChild(historyItem);
        });
    }

    // Dark mode implementation
    setupDarkMode();
    
    function setupDarkMode() {
        const darkModeToggle = document.getElementById('darkModeToggle');
        const darkIcon = document.getElementById('darkIcon');
        const lightIcon = document.getElementById('lightIcon');
        const html = document.documentElement;
        
        // Load user preference from localStorage
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        
        // Set initial state
        if (isDarkMode) {
            html.setAttribute('data-bs-theme', 'dark');
            darkIcon.classList.add('d-none');
            lightIcon.classList.remove('d-none');
        }
        
        // Toggle dark mode on button click
        darkModeToggle.addEventListener('click', function() {
            if (html.getAttribute('data-bs-theme') === 'dark') {
                html.setAttribute('data-bs-theme', 'light');
                darkIcon.classList.remove('d-none');
                lightIcon.classList.add('d-none');
                localStorage.setItem('darkMode', 'false');
            } else {
                html.setAttribute('data-bs-theme', 'dark');
                darkIcon.classList.add('d-none');
                lightIcon.classList.remove('d-none');
                localStorage.setItem('darkMode', 'true');
            }
        });
    }

    function generateMockMatches(winProb) {
        // Takım formuna göre son maçları simüle et (basit)
        const matches = [];
        for (let i = 0; i < 5; i++) {
            const rand = Math.random();
            if (rand < winProb) {
                matches.push('W');
            } else if (rand < winProb + 0.3) {
                matches.push('D');
            } else {
                matches.push('L');
            }
        }
        return matches;
    }
    
    function updateTeamComparison(comparison, homeTeam, awayTeam) {
        const comparisonSection = document.getElementById('teamComparisonSection');
        comparisonSection.style.display = 'block';
        
        // Takım isimleri
        document.getElementById('homeTeamNameComp').textContent = homeTeam;
        document.getElementById('awayTeamNameComp').textContent = awayTeam;
        
        const homeStats = comparison.home_team;
        const awayStats = comparison.away_team;
        
        // Form puanları
        const homeForm = document.getElementById('homeTeamForm');
        const awayForm = document.getElementById('awayTeamForm');
        homeForm.textContent = homeStats.form;
        awayForm.textContent = awayStats.form;
        
        // Form puanlarına göre renk atama
        updateStatColor(homeForm, awayForm, homeStats.form, awayStats.form);
        
        // Sezon puanları
        const homePoints = document.getElementById('homeTeamPoints');
        const awayPoints = document.getElementById('awayTeamPoints');
        homePoints.textContent = homeStats.season_points;
        awayPoints.textContent = awayStats.season_points;
        updateTextColor(homePoints, awayPoints, homeStats.season_points, awayStats.season_points);
        
        // Gol ortalamaları
        const homeGoals = document.getElementById('homeTeamGoalsScored');
        const awayGoals = document.getElementById('awayTeamGoalsScored');
        homeGoals.textContent = homeStats.goals_scored_avg.toFixed(1);
        awayGoals.textContent = awayStats.goals_scored_avg.toFixed(1);
        updateTextColor(homeGoals, awayGoals, homeStats.goals_scored_avg, awayStats.goals_scored_avg);
        
        // Yenilen gol ortalamaları (düşük olan daha iyi)
        const homeConceded = document.getElementById('homeTeamGoalsConceded');
        const awayConceded = document.getElementById('awayTeamGoalsConceded');
        homeConceded.textContent = homeStats.goals_conceded_avg.toFixed(1);
        awayConceded.textContent = awayStats.goals_conceded_avg.toFixed(1);
        updateTextColor(homeConceded, awayConceded, awayStats.goals_conceded_avg, homeStats.goals_conceded_avg); // Ters karşılaştırma
        
        // Son maçlar
        updateLastMatches('homeTeamLastMatches', homeStats.last_matches);
        updateLastMatches('awayTeamLastMatches', awayStats.last_matches);
    }
    
    function updateStatColor(homeElement, awayElement, homeValue, awayValue) {
        // Değerlere göre badge renklerini ayarla
        homeElement.className = 'badge';
        awayElement.className = 'badge';
        
        if (homeValue > awayValue) {
            homeElement.classList.add('bg-success');
            awayElement.classList.add(homeValue > awayValue * 1.5 ? 'bg-danger' : 'bg-warning');
        } else if (awayValue > homeValue) {
            awayElement.classList.add('bg-success');
            homeElement.classList.add(awayValue > homeValue * 1.5 ? 'bg-danger' : 'bg-warning');
        } else {
            homeElement.classList.add('bg-info');
            awayElement.classList.add('bg-info');
        }
    }
    
    function updateTextColor(homeElement, awayElement, homeValue, awayValue) {
        // Büyük olan değeri vurgula
        homeElement.classList.remove('text-success', 'text-danger');
        awayElement.classList.remove('text-success', 'text-danger');
        
        if (homeValue > awayValue) {
            homeElement.classList.add('text-success', 'fw-bold');
            awayElement.classList.add('text-danger');
        } else if (awayValue > homeValue) {
            awayElement.classList.add('text-success', 'fw-bold');
            homeElement.classList.add('text-danger');
        }
    }
    
    function updateLastMatches(containerId, matches) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        matches.forEach(result => {
            const badgeSpan = document.createElement('span');
            let badgeClass = 'badge ';
            
            switch (result) {
                case 'W':
                    badgeClass += 'bg-success';
                    break;
                case 'D':
                    badgeClass += 'bg-warning';
                    break;
                case 'L':
                    badgeClass += 'bg-danger';
                    break;
            }
            
            badgeSpan.className = badgeClass;
            badgeSpan.textContent = result;
            container.appendChild(badgeSpan);
        });
    }

    function startMatchSimulation() {
        // Daha önce bir simülasyon varsa temizle
        if (simulationInterval) {
            clearInterval(simulationInterval);
        }
        
        // Maç değişkenlerini sıfırla
        matchMinute = 0;
        homeScore = 0;
        awayScore = 0;
        homePossession = 50;
        
        // Maç olayları listesini temizle
        document.getElementById('matchEvents').innerHTML = '<li class="list-group-item p-2">Maç başladı!</li>';
        document.getElementById('liveScore').textContent = '0 - 0';
        
        // Butonları güncelle
        startSimulationBtn.style.display = 'none';
        stopSimulationBtn.style.display = 'inline-block';
        
        // Simülasyonu başlat (her 3 saniyede bir güncelle)
        simulationInterval = setInterval(updateMatchSimulation, 3000);
    }
    
    function stopMatchSimulation() {
        // Simülasyonu durdur
        if (simulationInterval) {
            clearInterval(simulationInterval);
            simulationInterval = null;
        }
        
        // Butonları güncelle
        startSimulationBtn.style.display = 'inline-block';
        stopSimulationBtn.style.display = 'none';
        
        // Maç sonlandı mesajı ekle
        addMatchEvent('Simülasyon durduruldu');
    }
    
    function updateMatchSimulation() {
        // Maç dakikasını artır
        matchMinute += 1;
        
        // 90 dakikaya ulaşıldığında simülasyonu durdur
        if (matchMinute > 90) {
            stopMatchSimulation();
            addMatchEvent('Maç sona erdi!');
            return;
        }
        
        // Maç zamanını güncelle
        document.getElementById('matchTime').textContent = matchMinute + "'";
        
        // Top hakimiyetini rastgele değiştir
        updatePossession();
        
        // Olayları rastgele oluştur
        generateRandomEvent();
    }
    
    function updatePossession() {
        // Top hakimiyetini her güncelleme için %5 içinde değiştir
        const possessionChange = Math.floor(Math.random() * 5) - 2;  // -2 ile +2 arası
        homePossession = Math.max(30, Math.min(70, homePossession + possessionChange));
        const awayPossession = 100 - homePossession;
        
        // Görsel güncellemeler
        document.getElementById('possessionBar').style.width = homePossession + '%';
        document.getElementById('possessionText').textContent = homePossession + '% - ' + awayPossession + '%';
    }
    
    function generateRandomEvent() {
        // Rastgele bir olay oluştur
        const eventTypes = [
            { type: 'shot', probability: 0.15 },
            { type: 'goal', probability: 0.03 },
            { type: 'corner', probability: 0.08 },
            { type: 'card', probability: 0.05 },
            { type: 'substitution', probability: matchMinute > 60 ? 0.06 : 0.02 }
        ];
        
        // Özel anlar: Maçın başı, devre arası, son dakikalar
        if (matchMinute === 1) {
            addMatchEvent('Maç başladı!');
            return;
        } else if (matchMinute === 45) {
            addMatchEvent('İlk yarı sona erdi');
            return;
        } else if (matchMinute === 46) {
            addMatchEvent('İkinci yarı başladı');
            return;
        }
        
        // Rastgele olay seçimi
        for (const eventType of eventTypes) {
            if (Math.random() < eventType.probability) {
                // Olayın ev sahibi için mi yoksa deplasman için mi olduğunu belirle
                const isHomeTeam = Math.random() < (homePossession / 100);
                const team = isHomeTeam ? currentHomeTeam : currentAwayTeam;
                
                switch (eventType.type) {
                    case 'shot':
                        addMatchEvent(`${team} atağında şut!`);
                        break;
                    case 'goal':
                        if (isHomeTeam) {
                            homeScore++;
                        } else {
                            awayScore++;
                        }
                        const liveScore = document.getElementById('liveScore');
                        liveScore.textContent = `${homeScore} - ${awayScore}`;
                        
                        // Gol animasyonu
                        liveScore.classList.add('score-changed');
                        setTimeout(() => liveScore.classList.remove('score-changed'), 1000);
                        
                        // Oyuncu seçimi
                        let goalScorer = "";
                        const players = isHomeTeam 
                            ? lastPredictions.goalscorer_predictions.home_team 
                            : lastPredictions.goalscorer_predictions.away_team;
                            
                        if (players && players.length > 0) {
                            // Oyuncuları gol atma olasılıklarına göre sırala
                            const sortedPlayers = [...players].sort((a, b) => b.scoring_prob - a.scoring_prob);
                            
                            // Ağırlıklı rastgele seçim - daha yüksek olasılıklı oyuncular daha sık seçilecek
                            const totalProb = sortedPlayers.reduce((sum, player) => sum + player.scoring_prob, 0);
                            let randomValue = Math.random() * totalProb;
                            let cumulativeProb = 0;
                            
                            for (const player of sortedPlayers) {
                                cumulativeProb += player.scoring_prob;
                                if (randomValue <= cumulativeProb) {
                                    goalScorer = player.name;
                                    break;
                                }
                            }
                        }
                        
                        // Gol mesajı
                        const goalMessage = goalScorer 
                            ? `⚽ GOL! ${team} ${matchMinute}' - ${goalScorer} - Yeni skor: ${homeScore}-${awayScore}`
                            : `⚽ GOL! ${team} ${matchMinute}' - Yeni skor: ${homeScore}-${awayScore}`;
                            
                        addMatchEvent(goalMessage, 'goal');
                        break;
                    case 'corner':
                        addMatchEvent(`${team} köşe vuruşu kazandı`);
                        break;
                    case 'card':
                        const isYellow = Math.random() < 0.8;
                        const cardIcon = isYellow ? '🟨' : '🟥';
                        addMatchEvent(`${cardIcon} ${isYellow ? 'Sarı' : 'Kırmızı'} kart ${team} oyuncusuna`, isYellow ? 'yellow-card' : 'red-card');
                        break;
                    case 'substitution':
                        addMatchEvent(`🔄 ${team} değişiklik yapıyor`);
                        break;
                }
                return;
            }
        }
    }
    
    function addMatchEvent(eventText, eventClass = '') {
        const eventsList = document.getElementById('matchEvents');
        const newEvent = document.createElement('li');
        newEvent.className = `list-group-item p-2 ${eventClass}`;
        newEvent.innerHTML = `<small class="text-muted">${matchMinute}'</small> ${eventText}`;
        
        // Yeni olay en üste eklensin
        eventsList.insertBefore(newEvent, eventsList.firstChild);
        
        // Liste çok uzarsa en alttaki olayları sil
        while (eventsList.children.length > 10) {
            eventsList.removeChild(eventsList.lastChild);
        }
    }
});

// Takım isimlerini otomatik tamamlama için daha kapsamlı veri
const teams = [
    // Türkiye Süper Lig
    'Galatasaray', 'Fenerbahçe', 'Beşiktaş', 'Trabzonspor',
    'Başakşehir', 'Alanyaspor', 'Sivasspor', 'Adana Demirspor',
    'Konyaspor', 'Antalyaspor', 'Kasımpaşa', 'Kayserispor',
    'Gaziantep FK', 'Hatayspor', 'Giresunspor', 'Samsunspor',
    'Pendikspor', 'İstanbulspor', 'Karagümrük', 'Ankaragücü',
    // İngiltere Premier Lig
    'Manchester City', 'Liverpool', 'Arsenal', 'Chelsea', 
    'Manchester United', 'Tottenham', 'Newcastle', 'Aston Villa',
    'Brighton', 'West Ham', 'Crystal Palace', 'Brentford',
    'Everton', 'Fulham', 'Wolverhampton', 'Bournemouth',
    'Nottingham Forest', 'Leicester City', 'Southampton', 'Ipswich',
    // İspanya La Liga
    'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Sevilla',
    'Villarreal', 'Real Sociedad', 'Real Betis', 'Athletic Bilbao',
    'Valencia', 'Espanyol', 'Mallorca', 'Celta Vigo',
    // Almanya Bundesliga
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
    'Wolfsburg', 'Eintracht Frankfurt', 'Borussia Mönchengladbach', 'Stuttgart',
    // İtalya Serie A
    'Inter Milan', 'AC Milan', 'Juventus', 'Napoli', 'Roma', 'Lazio',
    'Atalanta', 'Fiorentina', 'Bologna', 'Torino', 'Udinese', 'Sassuolo',
    // Fransa Ligue 1
    'PSG', 'Marseille', 'Lyon', 'Lille', 'Monaco', 'Nice', 'Lens', 'Rennes'
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