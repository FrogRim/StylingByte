using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class LoadJsonData : MonoBehaviour
{
    public void LoadJson()
    {
        string filePath = Path.Combine(Application.dataPath, "recommended_clothes_with_data.json");
        string jsonData = File.ReadAllText(filePath);

        // JSON 데이터를 배열 형식으로 변환
        ClothingItem[] itemsArray = JsonHelper.FromJson<ClothingItem>(jsonData);
        List<ClothingItem> clothingData = new List<ClothingItem>(itemsArray);

        foreach (ClothingItem item in clothingData)
        {
            Debug.Log("Product: " + item.product + " Prediction Score: " + item.prediction_score);
            foreach (string similar in item.most_similar)
            {
                Debug.Log("Similar Item: " + similar);
            }
        }
    }
}

[System.Serializable]
public class ClothingItem
{
    public string product;
    public List<string> most_similar;
    public float prediction_score;
}

public static class JsonHelper
{
    public static T[] FromJson<T>(string json)
    {
        string newJson = "{ \"array\": " + json + "}";
        Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(newJson);
        return wrapper.array;
    }

    [System.Serializable]
    private class Wrapper<T>
    {
        public T[] array;
    }
}
