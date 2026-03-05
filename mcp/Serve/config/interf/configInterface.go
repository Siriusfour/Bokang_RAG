package interf

import (
	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
	"log"
	"mcp/app/core/container"
	"mcp/global"
	"sync"
	"time"
)

type ConfigInterface interface {
	ConfigFileChangeListen()
	Clone(fileName string) ConfigInterface
	Get(keyName string) interface{}
	GetString(keyName string) string
	GetBool(keyName string) bool
	GetInt(keyName string) int
	GetInt32(keyName string) int32
	GetInt64(keyName string) int64
	GetFloat64(keyName string) float64
	GetDuration(keyName string) time.Duration
	GetStringSlice(keyName string) []string
}

type ConfigFile struct {
	Viper *viper.Viper
	Mu    *sync.Mutex
}

var configContainer = container.CreateContainersFactory()
var lastChangeTime = time.Now()

func (Config *ConfigFile) ConfigFileChangeListen() {

	Config.Viper.OnConfigChange(func(changeEvent fsnotify.Event) {
		if time.Now().Sub(lastChangeTime).Seconds() >= 1 {
			if changeEvent.Op.String() == "WRITE" {
				Config.clearCache()
				lastChangeTime = time.Now()
			}
		}
	})
	Config.Viper.WatchConfig()

}

func (Config *ConfigFile) Clone(fileName string) ConfigInterface {

	CloneConfigFile := *Config
	CloneConfigFile.Viper.AddConfigPath(global.BasePath + "/Config")
	CloneConfigFile.Viper.SetConfigName(fileName)
	CloneConfigFile.Viper.SetConfigType("yaml")
	CloneConfigFile.Mu = &sync.Mutex{}
	if err := CloneConfigFile.Viper.ReadInConfig(); err != nil {
		log.Fatal("clone Config is fail" + err.Error())
	}
	return &CloneConfigFile

}

func (Config *ConfigFile) Get(keyName string) interface{} {
	if Config.KeyIsExistsCache(keyName) {
		value := Config.GetCache(keyName)
		return value
	}

	value := Config.Viper.Get(keyName)
	Config.Cache(keyName, value)
	return value
}

// GetString 字符串格式返回值
func (Config *ConfigFile) GetString(keyName string) string {
	if Config.KeyIsExistsCache(keyName) {
		if s, ok := Config.GetCache(keyName).(string); ok {
			return s
		}

		return ""

	} else {
		value := Config.Viper.GetString(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetBool 布尔格式返回值
func (Config *ConfigFile) GetBool(keyName string) bool {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).(bool)
	} else {
		value := Config.Viper.GetBool(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetInt 整数格式返回值
func (Config *ConfigFile) GetInt(keyName string) int {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).(int)
	} else {
		value := Config.Viper.GetInt(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetInt32 整数格式返回值
func (Config *ConfigFile) GetInt32(keyName string) int32 {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).(int32)
	} else {
		value := Config.Viper.GetInt32(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetInt64 整数格式返回值
func (Config *ConfigFile) GetInt64(keyName string) int64 {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).(int64)
	} else {
		value := Config.Viper.GetInt64(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetFloat64 小数格式返回值
func (Config *ConfigFile) GetFloat64(keyName string) float64 {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).(float64)
	} else {
		value := Config.Viper.GetFloat64(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetDuration 时间单位格式返回值
func (Config *ConfigFile) GetDuration(keyName string) time.Duration {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).(time.Duration)
	} else {
		value := Config.Viper.GetDuration(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

// GetStringSlice 字符串切片数格式返回值
func (Config *ConfigFile) GetStringSlice(keyName string) []string {
	if Config.KeyIsExistsCache(keyName) {
		return Config.GetCache(keyName).([]string)
	} else {
		value := Config.Viper.GetStringSlice(keyName)
		Config.Cache(keyName, value)
		return value
	}
}

//=================================================

func (Config *ConfigFile) KeyIsExistsCache(keyName string) bool {

	_, ok := configContainer.KeyIsExists(global.ConfigKeyPrefix + keyName)

	if ok {
		return true
	}
	return false
}

func (Config *ConfigFile) GetCache(key string) interface{} {

	return configContainer.Get(global.ConfigKeyPrefix + key)

}

func (Config *ConfigFile) Cache(keyName string, value interface{}) bool {

	Config.Mu.Lock()
	defer Config.Mu.Unlock()

	//如果不存在于SMap，则调用set写入该key-value
	ok := Config.KeyIsExistsCache(keyName)
	if !ok {
		configContainer.Set(global.ConfigKeyPrefix+keyName, value)

	}
	return true
}

func (Config *ConfigFile) clearCache() {
	configContainer.FuzzyDelete(global.ConfigKeyPrefix)
}
