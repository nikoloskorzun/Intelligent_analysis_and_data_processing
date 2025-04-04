document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const inputs = {
        relative_compactness: parseFloat(document.getElementById('relative_compactness').value),
        surface_area: parseFloat(document.getElementById('surface_area').value),
        continuous_wall_area: parseFloat(document.getElementById('continuous_wall_area').value),
        roof_area: parseFloat(document.getElementById('roof_area').value),
        total_height: parseFloat(document.getElementById('total_height').value),
        orientation: parseFloat(document.getElementById('orientation').value),
        continuous_glazing_area: parseFloat(document.getElementById('continuous_glazing_area').value),
        glazing_area_distribution: parseFloat(document.getElementById('glazing_area_distribution').value)
    };

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputs)
        });

        const data = await response.json();
        document.getElementById('result').innerHTML = `Вычисленная отопительная нагрузка: ${data.prediction.toFixed(3)}`;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = 'Ошибка';
    }
});