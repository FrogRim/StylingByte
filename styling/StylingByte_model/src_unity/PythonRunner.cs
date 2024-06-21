using System;
using UnityEngine;
using System.Diagnostics;
using UnityEngine.UI;

public class PythonRunner : MonoBehaviour
{
    [SerializeField]
    public LoadJsonData loadJsonData;
    public Text Gender;
    public Text Season;
    public Text usage;
    public Text color;

    // Update is called once per frame
    public void recommend_start()
    {
        RunPythonScript();
        UnityEngine.Debug.Log("Python script is running.");
    }

    void RunPythonScript()
    {
        try
        {
            Process psi = new Process();
            // Python 실행 파일 경로 (예: C:\Python39\python.exe)
            psi.StartInfo.FileName = @"C:/Users/disse/AppData/Local/Programs/Python/Python312/python.exe";
            // 스크립트 파일 경로와 인자 추가
            string script = @"C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/ai_cody_result.py";
            string arguments = string.Format("\"{0}\" \"{1}\" \"{2}\" \"{3}\" \"{4}\"", script, Gender.text, Season.text, usage.text, color.text);
            psi.StartInfo.Arguments = arguments;
            psi.StartInfo.CreateNoWindow = true;
            psi.StartInfo.UseShellExecute = false;
            psi.StartInfo.RedirectStandardOutput = true;
            psi.StartInfo.RedirectStandardError = true;
            UnityEngine.Debug.Log("1. python");

            psi.Start();
            psi.WaitForExit();  // 프로세스가 완료될 때까지 대기

            // 로그 출력
            string output = psi.StandardOutput.ReadToEnd();
            string error = psi.StandardError.ReadToEnd();
            UnityEngine.Debug.Log("Python script output: " + output);
            if (!string.IsNullOrEmpty(error))
            {
                UnityEngine.Debug.LogError("Python script error: " + error);
            }

            UnityEngine.Debug.Log("Python script ended.");
            UnityEngine.Debug.Log("Print Result.");

            loadJsonData.LoadJson();
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("Unable to launch Python script: " + e.Message);
        }
    }
}
